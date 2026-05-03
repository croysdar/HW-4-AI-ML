"""
bnn_serengeti2.py
=================
BNN training & inference for Serengeti2 wildlife camera dataset.
Binary classification: non_blank (ANIMAL DETECTED) vs blank (EMPTY).

Architecture mirrors the hardware BNN Accelerator Chiplet design
(see algorithm_diagram.png and sw_baseline.md):

  Host CPU  : Data load · BatchNorm2d · Pooling · Linear (SW partition)
  Chiplet HW: BinarizeConv2d engine (3 layers) · XNOR+Popcount · spatial logic

  Conv2d(3→32,            3×3, stride=1, pad=1) → BN32          → [B,32,224,224]  8-bit
  BinarizeConv2d(32→64,  3×3, stride=2, pad=1) → BN64 → sign  → [B,64,112,112]  1-bit
  BinarizeConv2d(64→128, 3×3, stride=2, pad=1) → BN128 → sign → [B,128,56,56]   1-bit
  BinarizeConv2d(128→256,3×3, stride=2, pad=1) → BN256 → sign → [B,256,28,28]   1-bit
  AdaptiveAvgPool2d(1×1)                                        → [B,256]
  Linear(256→2)                                                 → logits

STE (Straight-Through Estimator): enables gradient flow through sign().
Weight clipping to [-1, 1] after each step stabilizes BNN training.

Usage:
  python bnn_serengeti2.py                         # train + demo
  confidence_check("path/to/image.jpg", model)     # from Jupyter
"""

import json
import os
import sys
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # macOS OpenMP conflict workaround

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
CHECKPOINT  = os.path.join(_SCRIPT_DIR, "bnn_serengeti2.pth")
_SEQ_DIR    = os.path.join(_SCRIPT_DIR, "data_sequences")
_BBOX_PATH  = os.path.join(_SCRIPT_DIR, "bbox_annotations.json")

RRR_LAMBDA  = 0.0   # weight of RRR spatial loss; 0.0 = disabled

BATCH_SIZE      = 32    # physical batch size (GPU memory)
ACCUM_STEPS     = 4     # gradient accumulation → effective batch = 32 × 4 = 128
EPOCHS          = 25
LR              = 7.64e-4  # Optuna best (trial 3, 85.6% in 10 epochs)
IMG_SIZE        = 224
EARLY_STOP_PAT  = 15   # stop if val acc doesn't improve for this many epochs
GRAD_CLIP       = 0.775  # Optuna best (trial 3)

# ImageFolder sorts classes alphabetically: blank=0, non_blank=1
_BLANK_IDX    = 0
_NONBLANK_IDX = 1


# ── STE Binarize ──────────────────────────────────────────────────────────────
class _STESign(torch.autograd.Function):
    """
    Forward : sign(x), mapping 0 → +1 to stay strictly in {+1, -1}.
    Backward: straight-through — passes grad where |x| ≤ 1, else zeroes it.
    This lets real-valued latent weights receive meaningful gradients even
    though the actual convolution operates on binarized values.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        b = torch.sign(x)
        return torch.where(b == 0, torch.ones_like(x), b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1.0).float()

binarize = _STESign.apply


# ── BinarizeConv2d ────────────────────────────────────────────────────────────
class BinarizeConv2d(nn.Conv2d):
    """
    Convolution layer that binarizes both weights and input activations.
    Simulates the XNOR+Popcount engine in the hardware chiplet.
    Latent real-valued weights are stored in self.weight and clipped to
    [-1, 1] after every optimizer step by the training loop.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_w = binarize(self.weight)
        binary_x = binarize(x)
        return F.conv2d(binary_x, binary_w, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


# ── BNN Classifier ────────────────────────────────────────────────────────────
class BNNClassifier(nn.Module):
    """Three-layer BNN matching the hardware accelerator chiplet design."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Conv1: 8-bit — standard conv preserves high-fidelity spatial features
        self.conv1 = nn.Conv2d(3,   32,  3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        # Conv2–4: 1-bit BNN — XNOR+Popcount engine on chiplet
        self.conv2 = BinarizeConv2d(32,  64,  3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = BinarizeConv2d(64,  128, 3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = BinarizeConv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))              # [B,32,224,224] — 8-bit, no binarize
        x = binarize(self.bn2(self.conv2(x)))    # [B,64,112,112] — 1-bit
        x = binarize(self.bn3(self.conv3(x)))    # [B,128,56,56]  — 1-bit
        x = binarize(self.bn4(self.conv4(x)))    # [B,256,28,28]  — 1-bit
        x = self.pool(x)                          # [B,256,1,1]
        x = torch.flatten(x, 1)                  # [B,256]
        return self.fc(x)                         # [B,2]


# ── Transforms ────────────────────────────────────────────────────────────────
_BORDER_PX = 5  # uniform border mask to remove camera edge artifacts (at 224×224)

class _MaskBanner(torch.nn.Module):
    """Zero a uniform border around the image to remove camera edge artifacts."""
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clone()
        t[:, :_BORDER_PX, :]  = 0.0  # top
        t[:, -_BORDER_PX:, :] = 0.0  # bottom
        t[:, :, :_BORDER_PX]  = 0.0  # left
        t[:, :, -_BORDER_PX:] = 0.0  # right
        return t

# Normalize to [-1, 1] so that sign() binarizes near the decision boundary.
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    _MaskBanner(),
])

# Training-only: augmentation improves generalization across lighting/scenes.
# Flip and slight colour jitter are safe for blank-vs-animal — they don't
# change whether an animal is present. No geometric distortions that could
# confuse the spatial structure the hardware relies on.
_train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Simulate IR/nighttime greyscale at low probability.
    # p=0.2 keeps most training in color while exposing the model to
    # monochrome empty scenes — the main gap in the night distribution.
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Randomly erase small patches (2–15% of image area) to prevent the model
    # from relying on any single texture region — directly targets the scattered
    # rock/shadow activations seen in hard-blank Grad-CAM heatmaps.
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    _MaskBanner(),
])


# ── Hard-blank sequence helpers ───────────────────────────────────────────────
def _load_hard_blank_frames(seq_dir: str = _SEQ_DIR) -> torch.Tensor | None:
    """Pre-load all blank sequence frames into a single tensor for fast per-epoch eval."""
    index_path = Path(seq_dir) / "seq_index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    frames = []
    for entry in index:
        if entry["label"] != "blank":
            continue
        seq_path = Path(seq_dir) / "blank" / f"seq_{entry['seq_idx']:05d}"
        for f in sorted(seq_path.glob("frame_*.jpg")):
            frames.append(_transform(Image.open(f).convert("RGB")))
    return torch.stack(frames) if frames else None  # [N, 3, 224, 224]


def _hard_blank_far(model: nn.Module, frames: torch.Tensor,
                    threshold: float = 0.5) -> float:
    """Per-frame FAR on the pre-loaded hard-blank tensor."""
    model.eval()
    fp = tn = 0
    with torch.no_grad():
        for i in range(0, len(frames), 64):
            probs = torch.softmax(model(frames[i:i+64].to(DEVICE)), dim=1)
            dets  = probs[:, _NONBLANK_IDX] >= threshold
            fp   += int(dets.sum())
            tn   += int((~dets).sum())
    return 100.0 * fp / (fp + tn) if (fp + tn) else 0.0


# ── RRR spatial loss ─────────────────────────────────────────────────────────
def _load_bboxes(path: str = _BBOX_PATH) -> dict:
    """Load bbox_annotations.json; returns {} if file missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class _StemSubset(torch.utils.data.Dataset):
    """Wraps an ImageFolder or Subset to also return the image stem."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if isinstance(self.ds, torch.utils.data.Subset):
            path = self.ds.dataset.imgs[self.ds.indices[idx]][0]
        else:
            path = self.ds.imgs[idx][0]
        return img, label, Path(path).stem


def _bbox_mask(boxes: list, feat_size: int = 56) -> torch.Tensor:
    """Build a binary [feat_size × feat_size] mask from a list of bbox dicts."""
    mask = torch.zeros(feat_size, feat_size)
    scale = feat_size / 224  # our stored images are 224×224
    for b in boxes:
        if not b.get("bbox"):
            continue
        x, y, w, h = b["bbox"]
        ow = b.get("orig_width")  or 224
        oh = b.get("orig_height") or 224
        # Scale: orig → 224 → feat_size
        sx, sy = 224 / ow * scale, 224 / oh * scale
        x1 = max(0, int(x * sx))
        y1 = max(0, int(y * sy))
        x2 = min(feat_size, int((x + w) * sx) + 1)
        y2 = min(feat_size, int((y + h) * sy) + 1)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


def _rrr_loss(feat_map: torch.Tensor, labels: torch.Tensor,
              stems: list[str], bboxes: dict) -> torch.Tensor:
    """
    Right-for-the-Right-Reasons penalty on bn3 feature maps.
    For each animal image in the batch that has a bbox annotation,
    penalise attention (mean-channel ReLU activation) outside the bbox.
    Returns a scalar loss (0 if no bbox-annotated animal images in batch).
    """
    penalties = []
    for i, (label, stem) in enumerate(zip(labels.tolist(), stems)):
        if label != _NONBLANK_IDX:
            continue
        info = bboxes.get(stem)
        if not info:
            continue
        boxes = [b for b in info["boxes"] if b.get("bbox")]
        if not boxes:
            continue
        mask = _bbox_mask(boxes, feat_size=feat_map.shape[-1]).to(feat_map.device)
        attn = F.relu(feat_map[i]).mean(0)          # [H, W]
        outside = (attn * (1.0 - mask)).sum()
        total   = attn.sum() + 1e-8
        penalties.append(outside / total)
    if not penalties:
        return feat_map.sum() * 0.0                 # zero with grad
    return torch.stack(penalties).mean()


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


def _filter_dataset(ds: datasets.ImageFolder, blacklist: set) -> torch.utils.data.Subset:
    if not blacklist:
        return ds
    keep = [i for i, (path, _) in enumerate(ds.imgs)
            if Path(path).stem not in blacklist]
    return torch.utils.data.Subset(ds, keep)


def make_loaders(data_root: str = DATA_ROOT):
    blacklist = _load_blacklist()
    if blacklist:
        print(f"Blacklist: {len(blacklist)} images excluded ({', '.join(sorted(blacklist))})")

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=_train_transform)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=_transform)
    train_ds = _filter_dataset(train_ds, blacklist)
    test_ds  = _filter_dataset(test_ds,  blacklist)

    print(f"Classes : {train_ds.dataset.classes if hasattr(train_ds, 'dataset') else train_ds.classes}  (blank=0, non_blank=1)")
    print(f"Train   : {len(train_ds):,} images")
    print(f"Test    : {len(test_ds):,}  images")
    kw = dict(batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True, pin_memory=False)
    return (
        DataLoader(_StemSubset(train_ds), shuffle=True,  **kw),
        DataLoader(test_ds,               shuffle=False, **kw),
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
                if   a and p:     tp += 1
                elif not a and not p: tn += 1
                elif not a and p: fp += 1
                else:             fn += 1
    n   = tp + tn + fp + fn
    acc = 100.0 * (tp + tn) / n
    return total_loss / n, acc, tp, tn, fp, fn


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(num_epochs: int = EPOCHS, data_root: str = DATA_ROOT,
          resume: bool = False, args_best_acc: float = 0.0,
          rrr_lambda: float = RRR_LAMBDA,
          warm_start: str | None = None,
          patience: int = EARLY_STOP_PAT) -> nn.Module:
    train_loader, test_loader = make_loaders(data_root)
    model     = BNNClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.00815)  # Optuna best
    # Upweight blank class to penalize false alarms — addresses night IR blank misclassification.
    # Weight [1.5, 1.0] means model pays 1.5× penalty for calling a blank image "animal".
    class_weights = torch.tensor([1.27, 1.0]).to(DEVICE)  # Optuna best
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    no_improve   = 0
    start_epoch  = 1

    if resume and os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        if "model" in ckpt:
            # New format — full restore
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_val_acc = ckpt["best_val_acc"]
            start_epoch  = ckpt["epoch"] + 1
            print(f"Resumed from epoch {ckpt['epoch']} — best val acc so far: {best_val_acc:.1f}%")
        else:
            # Old format (weights only) — load weights, fresh optimizer/scheduler
            model.load_state_dict(ckpt)
            best_val_acc = args_best_acc if args_best_acc > 0 else 0.0
            print("Loaded weights from checkpoint (old format — optimizer state unavailable).")
            print(f"Warm restart from trained weights. Will only save if val acc > {best_val_acc:.1f}%")
    elif resume:
        print("No checkpoint found — starting from scratch.")

    if warm_start and os.path.exists(warm_start):
        ckpt = torch.load(warm_start, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        if isinstance(ckpt, dict) and "best_val_acc" in ckpt:
            best_val_acc = ckpt["best_val_acc"]
        print(f"Warm-start weights loaded from {warm_start} "
              f"(must beat {best_val_acc:.1f}% to save)")

    hb_frames = _load_hard_blank_frames()
    if hb_frames is not None:
        print(f"  Hard-blank frames loaded: {len(hb_frames)} (from {_SEQ_DIR})")

    bboxes = _load_bboxes() if rrr_lambda > 0 else {}
    if bboxes:
        n_bbox = sum(1 for v in bboxes.values() if any(b.get("bbox") for b in v["boxes"]))
        print(f"  BBox annotations loaded: {n_bbox:,} images  (RRR λ={rrr_lambda})")

    hb_col  = "  HB-FAR" if hb_frames is not None else ""
    rrr_col = "  RRR" if rrr_lambda > 0 else ""
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'Recall':>7}  {'FAR':>6}{hb_col}{rrr_col}  {'Time':>6}  {'LR':>8}")
    print(f"  (effective batch size = {BATCH_SIZE} × {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS})")
    print("-" * (93 + (9 if hb_frames is not None else 0) + (6 if rrr_lambda > 0 else 0)))

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        optimizer.zero_grad()

        t0   = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}", unit="batch", leave=False, disable=not sys.stdout.isatty())
        epoch_rrr = 0.0
        for step, (imgs, labels, stems) in enumerate(pbar):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Register bn3 hook to capture feature maps for RRR loss
            _bn3_feat = {}
            if rrr_lambda > 0:
                def _fwd_hook(m, inp, out):
                    _bn3_feat["feat"] = out
                _hook_handle = model.bn3.register_forward_hook(_fwd_hook)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if rrr_lambda > 0:
                _hook_handle.remove()
                rrr = _rrr_loss(_bn3_feat["feat"], labels, stems, bboxes)
                loss = loss + rrr_lambda * rrr
                epoch_rrr += rrr.item()

            # Scale loss so accumulated gradients equal a true 128-image batch
            (loss / ACCUM_STEPS).backward()

            run_loss += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            n        += imgs.size(0)
            pbar.set_postfix(loss=f"{run_loss/n:.4f}", acc=f"{100.*correct/n:.1f}%")

            last_batch = (step + 1) == len(train_loader)
            if (step + 1) % ACCUM_STEPS == 0 or last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, BinarizeConv2d):
                            m.weight.data.clamp_(-1.0, 1.0)
                optimizer.zero_grad()

        t_loss     = run_loss / n
        t_acc      = 100.0 * correct / n
        v_loss, v_acc, vtp, vtn, vfp, vfn = evaluate(model, test_loader, criterion)
        recall     = 100.0 * vtp / (vtp + vfn) if (vtp + vfn) else 0.0
        far        = 100.0 * vfp / (vfp + vtn) if (vfp + vtn) else 0.0
        hb_far     = _hard_blank_far(model, hb_frames) if hb_frames is not None else None
        elapsed    = time.time() - t0
        epoch_time = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()   # cosine decay steps once per epoch

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            no_improve   = 0
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }, CHECKPOINT)
            marker = " ✓"
        else:
            no_improve += 1
            marker = ""

        hb_str  = f"  {hb_far:>5.1f}%" if hb_far is not None else ""
        rrr_str = f"  {epoch_rrr/len(train_loader):.3f}" if rrr_lambda > 0 else ""
        print(f"{epoch:>6}  {t_loss:>10.4f}  {t_acc:>8.1f}%  "
              f"{v_loss:>9.4f}  {v_acc:>7.1f}%  {recall:>6.1f}%  {far:>5.1f}%{hb_str}{rrr_str}  {epoch_time:>6}  {current_lr:>8.2e}{marker}")

        if no_improve >= patience:
            print(f"\nEarly stopping — val acc hasn't improved for {patience} epochs.")
            break

    print(f"\nBest val accuracy : {best_val_acc:.1f}%")
    print(f"Checkpoint saved  → {CHECKPOINT}")
    return model


# ── Test-Time Augmentation ────────────────────────────────────────────────────
def _tta_probs(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """
    Run 4 augmented views of tensor through model and return averaged softmax probs.
    Augmentations: original, horizontal flip, brightness +0.15, brightness -0.15.
    Tensor is already normalized to [-1, 1].
    """
    views = [
        tensor,
        torch.flip(tensor, dims=[3]),
        torch.clamp(tensor + 0.15, -1.0, 1.0),
        torch.clamp(tensor - 0.15, -1.0, 1.0),
    ]
    with torch.no_grad():
        probs = torch.stack([torch.softmax(model(v), dim=1) for v in views])
    return probs.mean(dim=0)  # [B, 2]


# ── Inference Helpers ─────────────────────────────────────────────────────────
def load_model(path: str = CHECKPOINT) -> nn.Module:
    """Load a saved BNNClassifier from a .pth checkpoint."""
    model = BNNClassifier()
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(DEVICE)
    model.eval()
    return model


def confidence_check(
    image_path: str,
    model: nn.Module = None,
    threshold: float = 0.5,
    tta: bool = False,
) -> tuple[str, float]:
    """
    Run inference on a single image and print the verdict.

    Args:
        image_path : Path to a .jpg (or any PIL-readable image).
        model      : BNNClassifier instance. Loads from CHECKPOINT if None.
        threshold  : Minimum p(animal) to call ANIMAL DETECTED (default 0.5).
        tta        : Enable test-time augmentation (4 views averaged, default False).

    Returns:
        (verdict, confidence_pct) — e.g. ("ANIMAL DETECTED", 87.3)
    """
    if model is None:
        model = load_model()

    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)   # [1,3,224,224]

    if tta:
        probs = _tta_probs(model, tensor)[0]            # [2]
    else:
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]  # [2]

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
        description="BNN Serengeti2 — wildlife camera classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  train                  Train the model and save a checkpoint.
  check <image.jpg> ...  Run inference on one or more images.

Examples:
  python bnn_serengeti2.py train
  python bnn_serengeti2.py train --epochs 20
  python bnn_serengeti2.py check photo.jpg
  python bnn_serengeti2.py check *.jpg
        """,
    )
    parser.add_argument("command", choices=["train", "check"],
                        help="'train' to train the model, 'check' to classify images")
    parser.add_argument("images", nargs="*", metavar="image.jpg",
                        help="Image path(s) to classify (used with 'check')")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--data-root", default=DATA_ROOT, metavar="DIR",
                        help=f"Dataset root containing train/ and test/ (default: {DATA_ROOT})")
    parser.add_argument("--threshold", type=float, default=0.5, metavar="T",
                        help="Min p(animal) to call ANIMAL DETECTED (default: 0.5)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation for 'check' (4 views averaged)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    parser.add_argument("--best-acc", type=float, default=0.0, metavar="ACC",
                        help="Known best val acc from old checkpoint (prevents overwriting with worse result)")
    parser.add_argument("--rrr-lambda", type=float, default=RRR_LAMBDA, metavar="λ",
                        help=f"Weight for RRR spatial loss (0=disabled, try 0.1–0.5, default: {RRR_LAMBDA})")
    parser.add_argument("--warm-start", default=None, metavar="CKPT",
                        help="Load model weights only from checkpoint (fresh optimizer/scheduler)")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PAT,
                        help=f"Early stopping patience in epochs (default: {EARLY_STOP_PAT})")
    args = parser.parse_args()

    if args.command == "train":
        print("\n" + "=" * 60)
        print("BNN Serengeti2 — Training")
        print("=" * 60)
        train(num_epochs=args.epochs, data_root=args.data_root,
              resume=args.resume, args_best_acc=args.best_acc,
              rrr_lambda=args.rrr_lambda, warm_start=args.warm_start,
              patience=args.patience)

    elif args.command == "check":
        if not args.images:
            parser.error("'check' requires at least one image path.\n"
                         "  Example: python bnn_serengeti2.py check photo.jpg")
        model = load_model()
        print(f"\n  Threshold: {args.threshold}  (raise to cut false alarms)")
        print(f"  TTA: {'enabled (4 views)' if args.tta else 'disabled'}")
        print(f"  {'Image':<40}  Result")
        print(f"  {'-'*40}  ------")
        for path in args.images:
            confidence_check(path, model, threshold=args.threshold, tta=args.tta)
