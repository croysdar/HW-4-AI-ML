"""
validate_precision.py
=====================
M2 Precision Validation: Compare FP32 Reference vs INT8/1-bit Hybrid DUT.

Reference model : BNNClassifier with Conv1 in full FP32 (as trained).
DUT             : Same checkpoint; Conv1 weights AND activations fake-quantized
                  to INT8 (symmetric per-tensor, [-128, 127]) before each forward
                  pass.  All other layers unchanged (Conv2–4 remain 1-bit XNOR).

Metrics reported:
  - Logit MAE and Max Error  (pre-softmax, pre-sigmoid output tensor comparison)
  - Top-1 Accuracy on 100 samples for both models
  - Accuracy Delta (DUT − Reference)

Usage:
  python project/m2/validate_precision.py
"""

import os
import sys
import copy

# Allow importing from project/ (bnn_serengeti2.py lives there)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from bnn_serengeti2 import (
    BNNClassifier,
    _transform,
    DEVICE,
    DATA_ROOT,
    binarize,
)

# Best trained checkpoint — FP32 baseline weights for both models
_CKPT = os.path.join(os.path.dirname(__file__), "..", "bnn_distilled_876pct.pth")
_N_SAMPLES = 100


# ── Fake INT8 Quantization ────────────────────────────────────────────────────

def fake_quantize_int8(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric per-tensor INT8 fake quantization.

    Simulates the pipeline an INT8 MAC unit would execute:
      1. Compute scale = max(|x|) / 127
      2. Quantize: round(x / scale), clamped to [-128, 127]
      3. Dequantize: multiply back by scale

    The result is FP32 in range but with INT8-precision granularity —
    exactly what "fake quantization" means in the QAT / PTQ literature.
    """
    scale = x.detach().abs().max().clamp(min=1e-8) / 127.0
    x_int = (x / scale).round().clamp(-128, 127)
    return x_int * scale


class _FakeQuantConv1(nn.Module):
    """Drop-in replacement for Conv1 that fake-quantizes both weights and input."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fq = fake_quantize_int8(self.conv.weight)
        x_fq = fake_quantize_int8(x)
        return F.conv2d(
            x_fq, w_fq, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


class HybridDUT(nn.Module):
    """
    BNNClassifier with Conv1 replaced by fake-INT8 quantization.
    Conv2–4 remain 1-bit XNOR (unchanged from the reference).
    All weights are shared with the reference via deepcopy at construction.
    """

    def __init__(self, reference: BNNClassifier):
        super().__init__()
        base = copy.deepcopy(reference)
        self.conv1 = _FakeQuantConv1(base.conv1)   # INT8 fake-quant
        self.bn1   = base.bn1
        self.conv2 = base.conv2                     # 1-bit XNOR
        self.bn2   = base.bn2
        self.conv3 = base.conv3                     # 1-bit XNOR
        self.bn3   = base.bn3
        self.conv4 = base.conv4                     # 1-bit XNOR
        self.bn4   = base.bn4
        self.pool  = base.pool
        self.fc    = base.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = binarize(self.bn2(self.conv2(x)))
        x = binarize(self.bn3(self.conv3(x)))
        x = binarize(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── Dataset Helpers ───────────────────────────────────────────────────────────

def _load_balanced_subset(data_root: str, n: int) -> Subset:
    """
    Return a Subset of exactly n images from the test set,
    drawn equally from each class (n//2 blank, n//2 non_blank).
    ImageFolder sorts classes alphabetically: blank=0, non_blank=1.
    """
    test_ds = datasets.ImageFolder(
        os.path.join(data_root, "test"), transform=_transform
    )
    per_class = n // 2
    class_indices = {c: [] for c in range(len(test_ds.classes))}
    for idx, (_, label) in enumerate(test_ds.imgs):
        class_indices[label].append(idx)

    selected = []
    for c in sorted(class_indices):
        selected.extend(class_indices[c][:per_class])
    selected.sort()
    return Subset(test_ds, selected)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device  : {DEVICE}")
    print(f"Samples : {_N_SAMPLES}  (balanced: {_N_SAMPLES//2} blank, {_N_SAMPLES//2} non_blank)")
    print(f"Checkpoint: {os.path.basename(_CKPT)}\n")

    # ── Load reference model ──────────────────────────────────────────────────
    reference = BNNClassifier().to(DEVICE)
    ckpt = torch.load(_CKPT, map_location=DEVICE, weights_only=False)
    reference.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    reference.eval()

    # ── Build DUT (same weights, Conv1 fake-quantized to INT8) ───────────────
    dut = HybridDUT(reference).to(DEVICE)
    dut.eval()

    # ── Load 100 balanced test images ─────────────────────────────────────────
    subset  = _load_balanced_subset(DATA_ROOT, _N_SAMPLES)
    loader  = DataLoader(subset, batch_size=_N_SAMPLES, shuffle=False, num_workers=0)
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

    # ── Forward passes ────────────────────────────────────────────────────────
    with torch.no_grad():
        ref_logits = reference(imgs)   # [N, 2]  FP32 reference logits
        dut_logits = dut(imgs)         # [N, 2]  INT8/1-bit DUT logits

    # ── Logit error metrics ───────────────────────────────────────────────────
    abs_err = (ref_logits - dut_logits).abs()
    mae     = abs_err.mean().item()
    max_err = abs_err.max().item()

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ref_preds = ref_logits.argmax(dim=1)
    dut_preds = dut_logits.argmax(dim=1)
    ref_acc   = (ref_preds == labels).float().mean().item() * 100.0
    dut_acc   = (dut_preds == labels).float().mean().item() * 100.0
    acc_delta = dut_acc - ref_acc

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 52)
    print("  Precision Validation Results  (N=100 samples)")
    print("=" * 52)
    print(f"  Reference (FP32 Conv1)  accuracy : {ref_acc:5.1f}%")
    print(f"  DUT (INT8/1-bit hybrid) accuracy : {dut_acc:5.1f}%")
    print(f"  Accuracy Delta (DUT − Ref)       : {acc_delta:+5.1f}%")
    print()
    print(f"  Logit MAE                        : {mae:.6f}")
    print(f"  Logit Max Error                  : {max_err:.6f}")
    print("=" * 52)
    print()
    print("Copy these numbers into project/m2/precision.md.")


if __name__ == "__main__":
    main()
