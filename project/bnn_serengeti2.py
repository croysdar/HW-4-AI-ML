"""
bnn_serengeti2.py
=================
BNN training & inference for Serengeti2 wildlife camera dataset.
Binary classification: non_blank (ANIMAL DETECTED) vs blank (EMPTY).

Architecture mirrors the hardware BNN Accelerator Chiplet design
(see algorithm_diagram.png and sw_baseline.md):

  Host CPU  : Data load · BatchNorm2d · Pooling · Linear (SW partition)
  Chiplet HW: BinarizeConv2d engine (3 layers) · XNOR+Popcount · spatial logic

  BinarizeConv2d(3→32,   3×3, stride=1, pad=1) → BN32 → sign  → [B,32,224,224]
  BinarizeConv2d(32→64,  3×3, stride=2, pad=1) → BN64 → sign  → [B,64,112,112]
  BinarizeConv2d(64→128, 3×3, stride=2, pad=1) → BN128 → sign → [B,128,56,56]
  AdaptiveAvgPool2d(1×1)                                        → [B,128]
  Linear(128→2)                                                 → logits

STE (Straight-Through Estimator): enables gradient flow through sign().
Weight clipping to [-1, 1] after each step stabilizes BNN training.

Usage:
  python bnn_serengeti2.py                         # train + demo
  confidence_check("path/to/image.jpg", model)     # from Jupyter
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # macOS OpenMP conflict workaround

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

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
DATA_ROOT   = os.path.join(_SCRIPT_DIR, "images", "archive")
CHECKPOINT  = os.path.join(_SCRIPT_DIR, "bnn_serengeti2.pth")

BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3
IMG_SIZE   = 224

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
        self.conv1 = BinarizeConv2d(3,   32,  3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = BinarizeConv2d(32,  64,  3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = BinarizeConv2d(64,  128, 3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = binarize(self.bn1(self.conv1(x)))   # [B,32,224,224]
        x = binarize(self.bn2(self.conv2(x)))   # [B,64,112,112]
        x = binarize(self.bn3(self.conv3(x)))   # [B,128,56,56]
        x = self.pool(x)                         # [B,128,1,1]
        x = torch.flatten(x, 1)                 # [B,128]
        return self.fc(x)                        # [B,2]


# ── Transforms ────────────────────────────────────────────────────────────────
# Normalize to [-1, 1] so that sign() binarizes near the decision boundary.
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ── Dataset helpers ───────────────────────────────────────────────────────────
def make_loaders(data_root: str = DATA_ROOT):
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=_transform)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=_transform)
    print(f"Classes : {train_ds.classes}  (blank=0, non_blank=1)")
    print(f"Train   : {len(train_ds):,} images")
    print(f"Test    : {len(test_ds):,}  images")
    # num_workers=0 avoids fork/MPS conflicts on macOS
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits       = model(imgs)
            total_loss  += criterion(logits, labels).item() * imgs.size(0)
            correct     += (logits.argmax(1) == labels).sum().item()
            n           += imgs.size(0)
    return total_loss / n, 100.0 * correct / n


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(num_epochs: int = EPOCHS, data_root: str = DATA_ROOT) -> nn.Module:
    train_loader, test_loader = make_loaders(data_root)
    model     = BNNClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}")
    print("-" * 55)

    for epoch in range(1, num_epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Clip latent weights to [-1, 1] — stabilizes BNN binary training
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, BinarizeConv2d):
                        m.weight.data.clamp_(-1.0, 1.0)

            run_loss += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            n        += imgs.size(0)

        t_loss = run_loss / n
        t_acc  = 100.0 * correct / n
        v_loss, v_acc = evaluate(model, test_loader, criterion)

        print(f"{epoch:>6}  {t_loss:>10.4f}  {t_acc:>8.1f}%  "
              f"{v_loss:>9.4f}  {v_acc:>7.1f}%")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nCheckpoint saved → {CHECKPOINT}")
    return model


# ── Inference Helpers ─────────────────────────────────────────────────────────
def load_model(path: str = CHECKPOINT) -> nn.Module:
    """Load a saved BNNClassifier from a .pth checkpoint."""
    model = BNNClassifier()
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def confidence_check(image_path: str, model: nn.Module = None) -> tuple[str, float]:
    """
    Run inference on a single image and print the verdict.

    Args:
        image_path : Path to a .jpg (or any PIL-readable image).
        model      : BNNClassifier instance. Loads from CHECKPOINT if None.

    Returns:
        (verdict, confidence_pct) — e.g. ("ANIMAL DETECTED", 87.3)

    Example (Jupyter):
        model = load_model()
        confidence_check("my_photo.jpg", model)
    """
    if model is None:
        model = load_model()

    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)   # [1,3,224,224]

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]  # [2]

    blank_p    = probs[_BLANK_IDX].item()
    nonblank_p = probs[_NONBLANK_IDX].item()

    if nonblank_p >= blank_p:
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
    args = parser.parse_args()

    if args.command == "train":
        print("\n" + "=" * 60)
        print("BNN Serengeti2 — Training")
        print("=" * 60)
        train(num_epochs=args.epochs)

    elif args.command == "check":
        if not args.images:
            parser.error("'check' requires at least one image path.\n"
                         "  Example: python bnn_serengeti2.py check photo.jpg")
        model = load_model()
        print(f"\n  {'Image':<40}  Result")
        print(f"  {'-'*40}  ------")
        for path in args.images:
            confidence_check(path, model)
