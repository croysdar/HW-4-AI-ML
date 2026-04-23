"""
distill.py
==========
Two-phase knowledge distillation for BNNClassifier.

Phase 1 — Train teacher:
  A pretrained ResNet-18 (ImageNet weights) is fine-tuned on the camera trap
  dataset. It acts as the "soft label" oracle for the student.

Phase 2 — Distill student:
  BNNClassifier is trained with a combined loss:
    L = alpha * KL(teacher_soft || student_soft) + (1 - alpha) * CE(student, hard)
  Temperature T softens the teacher distribution so the student learns richer
  inter-class signal beyond just 0/1 hard labels.

The resulting student has the *identical architecture* to a normally-trained
BNNClassifier — same XNOR operations, same chiplet partition, same power.
Only the training procedure differs.

Usage:
  # Step 1 — train teacher (fine-tune ResNet-18, ~20 epochs)
  python project/software_training/distill.py teacher \\
      --data-root project/data_combined --epochs 20

  # Step 2 — distill student (BNNClassifier, ~50 epochs)
  python project/software_training/distill.py student \\
      --data-root project/data_combined --epochs 50

  # Resume student distillation
  python project/software_training/distill.py student \\
      --data-root project/data_combined --epochs 50 --resume
"""

import os
import sys
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from bnn_serengeti2 import (
    BNNClassifier, make_loaders, evaluate,
    DEVICE, BATCH_SIZE, ACCUM_STEPS, BinarizeConv2d,
    LR, GRAD_CLIP,
)

_SCRIPT_DIR     = Path(__file__).parent.parent   # project/
TEACHER_CKPT    = _SCRIPT_DIR / "bnn_teacher.pth"
STUDENT_CKPT    = _SCRIPT_DIR / "bnn_distilled.pth"

# Distillation hyperparameters
TEMPERATURE  = 4.0   # soften teacher distribution; higher = softer
ALPHA        = 0.7   # weight on KL loss; 1-alpha on hard CE loss
BLANK_WEIGHT = 1.27  # Optuna best — applied to hard CE term only

EARLY_STOP_PAT = 7


# ── Teacher model ─────────────────────────────────────────────────────────────
def make_teacher(num_classes: int = 2) -> nn.Module:
    """ResNet-18 with ImageNet weights, final FC replaced for binary task."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Distillation loss ─────────────────────────────────────────────────────────
def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    T: float = TEMPERATURE,
    alpha: float = ALPHA,
) -> torch.Tensor:
    soft_teacher = F.softmax(teacher_logits / T, dim=1).detach()
    soft_student = F.log_softmax(student_logits / T, dim=1)
    # KL divergence scaled by T² to preserve gradient magnitude
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(student_logits, labels, weight=class_weights)
    return alpha * kl + (1.0 - alpha) * ce


# ── Shared transform (same as bnn_serengeti2 train augmentation) ──────────────
_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _make_loaders_custom(data_root: str):
    train_dir = Path(data_root) / "train"
    test_dir  = Path(data_root) / "test"
    train_ds  = datasets.ImageFolder(str(train_dir), transform=_train_transform)
    val_ds    = datasets.ImageFolder(str(test_dir),  transform=_val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


# ── Phase 1: Train teacher ────────────────────────────────────────────────────
def train_teacher(data_root: str, num_epochs: int):
    print(f"\n{'='*60}")
    print("PHASE 1 — Teacher Training (ResNet-18 fine-tune)")
    print(f"{'='*60}")
    print(f"Device   : {DEVICE}")
    print(f"Epochs   : {num_epochs}")
    print(f"Dataset  : {data_root}")
    print(f"Output   : {TEACHER_CKPT}\n")

    train_loader, val_loader = _make_loaders_custom(data_root)

    model     = make_teacher().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([BLANK_WEIGHT, 1.0]).to(DEVICE)
    )

    best_val_acc = 0.0
    no_improve   = 0

    print(f"  {'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'Time':>6}")
    print(f"  {'─'*50}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = total_correct = total_n = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss    += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n       += len(labels)

        tr_loss = total_loss / total_n
        tr_acc  = 100.0 * total_correct / total_n

        model.eval()
        val_loss = val_correct = val_n = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits    = model(imgs)
                val_loss += criterion(logits, labels).item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_n += len(labels)

        va_loss = val_loss / val_n
        va_acc  = 100.0 * val_correct / val_n
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        marker = " ✓" if va_acc > best_val_acc else ""

        print(f"  {epoch:>3}  {tr_loss:>7.4f}  {tr_acc:>5.1f}%  "
              f"{va_loss:>7.4f}  {va_acc:>5.1f}%  {mins}m{secs:02d}s{marker}")

        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve   = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "best_val_acc": best_val_acc}, TEACHER_CKPT)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"\nEarly stopping — no improvement for {EARLY_STOP_PAT} epochs.")
                break

    print(f"\nTeacher best val accuracy: {best_val_acc:.1f}%")
    print(f"Saved → {TEACHER_CKPT}")


# ── Phase 2: Distill student ──────────────────────────────────────────────────
def train_student(data_root: str, num_epochs: int, resume: bool = False):
    print(f"\n{'='*60}")
    print("PHASE 2 — Student Distillation (BNNClassifier)")
    print(f"{'='*60}")
    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {num_epochs}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Alpha       : {ALPHA}  (KL weight; {1-ALPHA:.1f} on hard CE)")
    print(f"Dataset     : {data_root}")
    print(f"Teacher     : {TEACHER_CKPT}")
    print(f"Output      : {STUDENT_CKPT}\n")

    if not TEACHER_CKPT.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_CKPT}\n"
            "Run `distill.py teacher` first."
        )

    # Load teacher — frozen, eval mode only
    teacher = make_teacher().to(DEVICE)
    t_ckpt  = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=True)
    teacher.load_state_dict(t_ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded  (best val acc: {t_ckpt['best_val_acc']:.1f}%)\n")

    # Student
    student   = BNNClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.00815)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    class_weights = torch.tensor([BLANK_WEIGHT, 1.0]).to(DEVICE)

    start_epoch  = 1
    best_val_acc = 0.0
    no_improve   = 0

    if resume and STUDENT_CKPT.exists():
        ckpt = torch.load(STUDENT_CKPT, map_location=DEVICE, weights_only=True)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val acc: {best_val_acc:.1f}%)\n")

    train_loader, val_loader = _make_loaders_custom(data_root)

    print(f"  {'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'Recall':>7}  {'FAR':>6}  {'Time':>6}")
    print(f"  {'─'*72}")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        student.train()
        optimizer.zero_grad()
        total_loss = total_correct = total_n = 0

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            student_logits = student(imgs)
            loss = distill_loss(student_logits, teacher_logits, labels, class_weights)
            (loss / ACCUM_STEPS).backward()

            last_batch = (step + 1) == len(train_loader)
            if (step + 1) % ACCUM_STEPS == 0 or last_batch:
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optimizer.step()
                with torch.no_grad():
                    for m in student.modules():
                        if isinstance(m, BinarizeConv2d):
                            m.weight.data.clamp_(-1.0, 1.0)
                optimizer.zero_grad()

            total_loss    += loss.item() * len(labels)
            total_correct += (student_logits.argmax(1) == labels).sum().item()
            total_n       += len(labels)

        tr_loss = total_loss / total_n
        tr_acc  = 100.0 * total_correct / total_n

        # Reuse evaluate() from bnn_serengeti2 for val metrics
        val_criterion = nn.CrossEntropyLoss(weight=class_weights)
        val_loss, va_acc, recall, far, _, _ = evaluate(student, val_loader, val_criterion)

        scheduler.step()

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        marker = " ✓" if va_acc > best_val_acc else ""

        print(f"  {epoch:>3}  {tr_loss:>7.4f}  {tr_acc:>5.1f}%  "
              f"{val_loss:>7.4f}  {va_acc:>5.1f}%  {recall:>6.1f}%  {far:>5.1f}%  "
              f"{mins}m{secs:02d}s{marker}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve   = 0
            torch.save({
                "epoch":        epoch,
                "model":        student.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }, STUDENT_CKPT)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"\nEarly stopping — val acc hasn't improved for {EARLY_STOP_PAT} epochs.")
                break

    print(f"\nDistilled student best val accuracy: {best_val_acc:.1f}%")
    print(f"Saved → {STUDENT_CKPT}")
    print(f"\nTo evaluate: python project/software_training/evaluate_bnn.py "
          f"--data-root project/data_combined --checkpoint {STUDENT_CKPT}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: ResNet-18 teacher → BNNClassifier student",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("phase", choices=["teacher", "student"],
                        help="'teacher' to fine-tune ResNet-18; 'student' to distill BNNClassifier")
    parser.add_argument("--data-root", default="project/data_combined", metavar="DIR")
    parser.add_argument("--epochs",    type=int, default=None,
                        help="Training epochs (default: 20 for teacher, 50 for student)")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume student from existing checkpoint (student phase only)")
    args = parser.parse_args()

    if args.phase == "teacher":
        train_teacher(args.data_root, args.epochs or 20)
    else:
        train_student(args.data_root, args.epochs or 50, resume=args.resume)


if __name__ == "__main__":
    main()
