"""
fp4_serengeti2.py
=================
Simulated-FP4 network for Serengeti2 wildlife camera dataset.
Binary classification: non_blank (ANIMAL DETECTED) vs blank (EMPTY).

Architecture mirrors BNNClassifier exactly — same layer sizes, same data —
so that accuracy differences isolate the effect of precision alone.

  Conv2d(3→32,            3×3, stride=1, pad=1) → BN32           [B,32,224,224]  FP32
  FP4Conv2d(32→64,  3×3, stride=2, pad=1) → BN64 → fp4_act     [B,64,112,112]  FP4
  FP4Conv2d(64→128, 3×3, stride=2, pad=1) → BN128 → fp4_act    [B,128,56,56]   FP4
  FP4Conv2d(128→256,3×3, stride=2, pad=1) → BN256 → fp4_act    [B,256,28,28]   FP4
  AdaptiveAvgPool2d(1×1)                                          [B,256]
  Linear(256→2)                                                   logits

FP4 format: IEEE-like E2M1 (2 exponent bits, 1 mantissa bit, bias=1).
Positive representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
Signed set (15 values): {±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0, 0}.

Quantization is simulated via STE:
  - Weights: per-output-channel absmax scaling, rounded to nearest FP4.
  - Activations: per-tensor absmax scaling, rounded to nearest FP4.
  - Accumulation and BatchNorm remain in FP32 (wider accumulator, as in hardware).

Usage:
  python fp4_serengeti2.py train
  python fp4_serengeti2.py check <image.jpg> ...
"""

import os
import sys
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Device: Apple MPS (M-series GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Device: CPU")

# ── Config ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(_SCRIPT_DIR, "data_20k")
CHECKPOINT  = os.path.join(_SCRIPT_DIR, "fp4_alpha_serengeti2.pth")

BATCH_SIZE      = 32
ACCUM_STEPS     = 4
EPOCHS          = 25
LR              = 7.64e-4
IMG_SIZE        = 224
EARLY_STOP_PAT  = 15
GRAD_CLIP       = 0.775

_BLANK_IDX    = 0
_NONBLANK_IDX = 1


# ── FP4 E2M1 Quantization ─────────────────────────────────────────────────────
# Positive FP4 values: subnormal {0, 0.5} and normal {1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
_FP4_POS    = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])
# All 15 signed values (excluding -0), ascending order
_FP4_SIGNED = torch.cat([-_FP4_POS[1:].flip(0), _FP4_POS])


def _round_to_fp4(x_norm: torch.Tensor) -> torch.Tensor:
    """Round each element of x_norm (in FP4 range [-6,6]) to nearest FP4 value.

    Processes in chunks of 4M elements: each chunk builds a [chunk,15] distance
    matrix (one argmin op) instead of 15 sequential tensor passes. ~10× fewer
    MPS kernel dispatches than the iterative approach, with bounded ~240MB peak.
    """
    vals  = _FP4_SIGNED.to(x_norm.device)
    flat  = x_norm.flatten()
    out   = torch.empty_like(flat)
    CHUNK = 4_000_000
    for i in range(0, len(flat), CHUNK):
        seg         = flat[i : i + CHUNK]
        dists       = (seg.unsqueeze(-1) - vals).abs()  # [chunk, 15]
        out[i : i + CHUNK] = vals[dists.argmin(-1)]
    return out.view_as(x_norm)


class _STEFP4Weight(torch.autograd.Function):
    """
    Forward : per-output-channel absmax → normalize to [-6,6] → round to FP4 → unscale.
    Backward: straight-through estimator (pass gradient unchanged).
    """
    @staticmethod
    def forward(_ctx, w: torch.Tensor) -> torch.Tensor:
        scale = (w.abs().view(w.shape[0], -1).max(dim=1).values / 6.0).clamp(min=1e-8)
        scale = scale.view(-1, *([1] * (w.dim() - 1)))
        return _round_to_fp4(w / scale) * scale

    @staticmethod
    def backward(_ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _STEFP4Act(torch.autograd.Function):
    """
    Forward : per-tensor absmax → normalize to [-6,6] → round to FP4 → unscale.
    Backward: straight-through estimator.
    """
    @staticmethod
    def forward(_ctx, x: torch.Tensor) -> torch.Tensor:
        scale = (x.abs().max() / 6.0).clamp(min=1e-8)
        return _round_to_fp4(x / scale) * scale

    @staticmethod
    def backward(_ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


fp4_weight = _STEFP4Weight.apply
fp4_act    = _STEFP4Act.apply


# ── FP4Conv2d ─────────────────────────────────────────────────────────────────
class FP4Conv2d(nn.Conv2d):
    """
    Conv layer with FP4-quantized weights and a learned per-channel scale (alpha),
    matching the XNOR-Net alpha used in BinarizeConv2d. Alpha recovers the relative
    channel magnitude information lost when all channels are normalized to [-6,6]
    during weight quantization.
    Activation quantization is applied externally in FP4Classifier.forward.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.ones(self.out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(
            x,
            fp4_weight(self.weight),
            self.bias, self.stride, self.padding, self.dilation, self.groups,
        )
        return out * self.alpha.view(1, -1, 1, 1)


# ── FP4 Classifier ────────────────────────────────────────────────────────────
class FP4Classifier(nn.Module):
    """
    Same layer topology as BNNClassifier; fp4_act replaces binarize as activation.
    First conv is kept in FP32 (matching the BNN's 8-bit first layer).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = FP4Conv2d(32,  64,  3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = FP4Conv2d(64,  128, 3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = FP4Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))           # [B,32,224,224] — FP32, no quantize
        x = fp4_act(self.bn2(self.conv2(x)))  # [B,64,112,112] — FP4
        x = fp4_act(self.bn3(self.conv3(x)))  # [B,128,56,56]  — FP4
        x = fp4_act(self.bn4(self.conv4(x)))  # [B,256,28,28]  — FP4
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── Transforms ────────────────────────────────────────────────────────────────
_BORDER_PX = 5

class _MaskBanner(torch.nn.Module):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clone()
        t[:, :_BORDER_PX, :]  = 0.0
        t[:, -_BORDER_PX:, :] = 0.0
        t[:, :, :_BORDER_PX]  = 0.0
        t[:, :, -_BORDER_PX:] = 0.0
        return t

# Normalize to [-1,1]: sign() lives at 0; FP4 subnormals at ±0.5 are also close.
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    _MaskBanner(),
])

_train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    _MaskBanner(),
])


# ── Dataset helpers ───────────────────────────────────────────────────────────
_BLACKLIST_PATH = os.path.join(_SCRIPT_DIR, "blacklist.txt")

def _load_blacklist() -> set:
    if not os.path.exists(_BLACKLIST_PATH):
        return set()
    stems = set()
    with open(_BLACKLIST_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                stems.add(line)
    return stems


def make_loaders(data_root: str = DATA_ROOT):
    blacklist = _load_blacklist()
    if blacklist:
        print(f"Blacklist: {len(blacklist)} images excluded")

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=_train_transform)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=_transform)

    if blacklist:
        keep_tr = [i for i, (p, _) in enumerate(train_ds.imgs) if Path(p).stem not in blacklist]
        keep_te = [i for i, (p, _) in enumerate(test_ds.imgs)  if Path(p).stem not in blacklist]
        train_ds = torch.utils.data.Subset(train_ds, keep_tr)
        test_ds  = torch.utils.data.Subset(test_ds,  keep_te)

    base_tr = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    print(f"Classes : {base_tr.classes}  (blank=0, non_blank=1)")
    print(f"Train   : {len(train_ds):,} images")
    print(f"Test    : {len(test_ds):,}  images")
    kw = dict(batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             threshold: float = 0.5):
    """Returns (loss, accuracy, tp, tn, fp, fn)."""
    model.eval()
    total_loss = 0.0
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            probs        = torch.softmax(model(imgs), dim=1)
            preds        = (probs[:, _NONBLANK_IDX] >= threshold).long()
            total_loss  += criterion(model(imgs), labels).item() * imgs.size(0)
            for pred, label in zip(preds.tolist(), labels.tolist()):
                a = (label == _NONBLANK_IDX)
                p = (pred  == _NONBLANK_IDX)
                if   a and p:         tp += 1
                elif not a and not p: tn += 1
                elif not a and p:     fp += 1
                else:                 fn += 1
    n   = tp + tn + fp + fn
    acc = 100.0 * (tp + tn) / n
    return total_loss / n, acc, tp, tn, fp, fn


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(num_epochs: int = EPOCHS, data_root: str = DATA_ROOT,
          resume: bool = False, args_best_acc: float = 0.0,
          warm_start: str | None = None,
          patience: int = EARLY_STOP_PAT,
          checkpoint: str = CHECKPOINT,
          lr: float = LR,
          weight_decay: float = 0.00815,
          blank_weight: float = 1.27,
          grad_clip: float = GRAD_CLIP) -> nn.Module:

    train_loader, test_loader = make_loaders(data_root)
    model     = FP4Classifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = torch.tensor([blank_weight, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    no_improve   = 0
    start_epoch  = 1

    if resume and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_val_acc = ckpt["best_val_acc"]
            start_epoch  = ckpt["epoch"] + 1
            print(f"Resumed from epoch {ckpt['epoch']} — best val acc so far: {best_val_acc:.1f}%")
        else:
            model.load_state_dict(ckpt, strict=False)
            best_val_acc = args_best_acc if args_best_acc > 0 else 0.0
            print("Loaded weights (old format — fresh optimizer/scheduler).")
    elif resume:
        print("No checkpoint found — starting from scratch.")

    if warm_start and os.path.exists(warm_start):
        ckpt = torch.load(warm_start, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        if args_best_acc > 0:
            best_val_acc = args_best_acc
        elif isinstance(ckpt, dict) and "best_val_acc" in ckpt:
            best_val_acc = ckpt["best_val_acc"]
        print(f"Warm-start weights loaded from {warm_start} "
              f"(must beat {best_val_acc:.1f}% to save)")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'Recall':>7}  {'FAR':>6}  {'Time':>6}  {'LR':>8}")
    print(f"  (effective batch size = {BATCH_SIZE} × {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS})")
    print("-" * 93)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        optimizer.zero_grad()

        t0   = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}", unit="batch",
                    leave=False, disable=not sys.stdout.isatty())

        for step, batch in enumerate(pbar):
            imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            logits = model(imgs)
            loss   = criterion(logits, labels)
            (loss / ACCUM_STEPS).backward()

            run_loss += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            n        += imgs.size(0)
            pbar.set_postfix(loss=f"{run_loss/n:.4f}", acc=f"{100.*correct/n:.1f}%")

            last_batch = (step + 1) == len(train_loader)
            if (step + 1) % ACCUM_STEPS == 0 or last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        t_loss     = run_loss / n
        t_acc      = 100.0 * correct / n
        v_loss, v_acc, vtp, vtn, vfp, vfn = evaluate(model, test_loader, criterion)
        recall     = 100.0 * vtp / (vtp + vfn) if (vtp + vfn) else 0.0
        far        = 100.0 * vfp / (vfp + vtn) if (vfp + vtn) else 0.0
        elapsed    = time.time() - t0
        epoch_time = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            no_improve   = 0
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }, checkpoint)
            marker = " ✓"
        else:
            no_improve += 1
            marker = ""

        print(f"{epoch:>6}  {t_loss:>10.4f}  {t_acc:>8.1f}%  "
              f"{v_loss:>9.4f}  {v_acc:>7.1f}%  {recall:>6.1f}%  {far:>5.1f}%  "
              f"{epoch_time:>6}  {current_lr:>8.2e}{marker}")

        if no_improve >= patience:
            print(f"\nEarly stopping — val acc hasn't improved for {patience} epochs.")
            break

    print(f"\nBest val accuracy : {best_val_acc:.1f}%")
    print(f"Checkpoint saved  → {checkpoint}")
    return model


# ── Test-Time Augmentation ────────────────────────────────────────────────────
def _tta_probs(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    views = [
        tensor,
        torch.flip(tensor, dims=[3]),
        torch.clamp(tensor + 0.15, -1.0, 1.0),
        torch.clamp(tensor - 0.15, -1.0, 1.0),
    ]
    with torch.no_grad():
        probs = torch.stack([torch.softmax(model(v), dim=1) for v in views])
    return probs.mean(dim=0)


# ── Inference Helpers ─────────────────────────────────────────────────────────
def load_model(path: str = CHECKPOINT) -> nn.Module:
    model = FP4Classifier()
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def confidence_check(
    image_path: str,
    model: nn.Module = None,
    threshold: float = 0.5,
    tta: bool = False,
) -> tuple[str, float]:
    if model is None:
        model = load_model()

    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    if tta:
        probs = _tta_probs(model, tensor)[0]
    else:
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]

    nonblank_p = probs[_NONBLANK_IDX].item()
    blank_p    = probs[_BLANK_IDX].item()

    if nonblank_p >= threshold:
        verdict, confidence = "ANIMAL DETECTED", nonblank_p * 100
    else:
        verdict, confidence = "EMPTY", blank_p * 100

    print(f"  {os.path.basename(image_path):<40}  →  {verdict}  ({confidence:.1f}%)")
    return verdict, confidence


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FP4 Serengeti2 — simulated FP4 wildlife camera classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  train                  Train the model and save a checkpoint.
  check <image.jpg> ...  Run inference on one or more images.

Examples:
  python fp4_serengeti2.py train
  python fp4_serengeti2.py train --epochs 20
  python fp4_serengeti2.py check photo.jpg
        """,
    )
    parser.add_argument("command", choices=["train", "check"])
    parser.add_argument("images", nargs="*", metavar="image.jpg")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--data-root",    default=DATA_ROOT, metavar="DIR")
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--tta",          action="store_true")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--best-acc",     type=float, default=0.0)
    parser.add_argument("--warm-start",   default=None, metavar="CKPT")
    parser.add_argument("--patience",     type=int,   default=EARLY_STOP_PAT)
    parser.add_argument("--checkpoint",   default=None, metavar="PATH")
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=0.00815)
    parser.add_argument("--blank-weight", type=float, default=1.27)
    parser.add_argument("--grad-clip",    type=float, default=GRAD_CLIP)
    args = parser.parse_args()

    ckpt_path = (os.path.join(_SCRIPT_DIR, args.checkpoint)
                 if args.checkpoint and not os.path.isabs(args.checkpoint)
                 else args.checkpoint or CHECKPOINT)

    if args.command == "train":
        print("\n" + "=" * 60)
        print("FP4 Serengeti2 — Training (simulated FP4 E2M1)")
        print("=" * 60)
        train(num_epochs=args.epochs, data_root=args.data_root,
              resume=args.resume, args_best_acc=args.best_acc,
              warm_start=args.warm_start, patience=args.patience,
              checkpoint=ckpt_path, lr=args.lr,
              weight_decay=args.weight_decay, blank_weight=args.blank_weight,
              grad_clip=args.grad_clip)

    elif args.command == "check":
        if not args.images:
            parser.error("'check' requires at least one image path.")
        model = load_model(ckpt_path)
        print(f"\n  Threshold: {args.threshold}")
        print(f"  TTA: {'enabled (4 views)' if args.tta else 'disabled'}")
        print(f"  {'Image':<40}  Result")
        print(f"  {'-'*40}  ------")
        for path in args.images:
            confidence_check(path, model, threshold=args.threshold, tta=args.tta)
