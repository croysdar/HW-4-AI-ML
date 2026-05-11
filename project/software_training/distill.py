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
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from bnn_serengeti2 import (
    BNNClassifier, evaluate, make_loaders,
    _train_transform, _transform,
    DEVICE, BATCH_SIZE, ACCUM_STEPS, BinarizeConv2d,
    LR, GRAD_CLIP,
)

_SCRIPT_DIR     = Path(__file__).parent.parent   # project/
TEACHER_CKPT    = _SCRIPT_DIR / "bnn_teacher.pth"
STUDENT_CKPT    = _SCRIPT_DIR / "bnn_distilled.pth"

# Distillation hyperparameters
TEMPERATURE  = 2.0   # soften teacher distribution; higher = softer
ALPHA        = 0.3   # weight on KL loss; 1-alpha on hard CE loss
BLANK_WEIGHT = 1.27  # Optuna best — applied to hard CE term only

EARLY_STOP_PAT         = 7
EARLY_STOP_PAT_TEACHER = 12   # teacher needs more runway


# ── Teacher model ─────────────────────────────────────────────────────────────
def make_teacher(num_classes: int = 2) -> nn.Module:
    """ResNet-50 with ImageNet weights, final FC replaced for binary task."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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




# ── Phase 1: Train teacher ────────────────────────────────────────────────────
def train_teacher(data_root: str, num_epochs: int,
                  teacher_ckpt: Path = TEACHER_CKPT,
                  blank_weight: float = BLANK_WEIGHT):
    print(f"\n{'='*60}")
    print("PHASE 1 — Teacher Training (ResNet-50 fine-tune)")
    print(f"{'='*60}")
    print(f"Device   : {DEVICE}")
    print(f"Epochs   : {num_epochs}")
    print(f"Dataset  : {data_root}")
    print(f"Output   : {teacher_ckpt}\n")

    train_loader, val_loader = make_loaders(data_root)

    model     = make_teacher().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([blank_weight, 1.0]).to(DEVICE)
    )

    best_val_acc = 0.0
    no_improve   = 0

    print(f"  {'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'Recall':>7}  {'FAR':>6}  {'Time':>6}")
    print(f"  {'─'*72}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = total_correct = total_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}", unit="batch",
                    leave=False, disable=not sys.stdout.isatty())
        for imgs, labels, _ in pbar:
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
            pbar.set_postfix(loss=f"{total_loss/total_n:.4f}", acc=f"{100.*total_correct/total_n:.1f}%")

        tr_loss = total_loss / total_n
        tr_acc  = 100.0 * total_correct / total_n

        va_loss, va_acc, tp, tn, fp, fn = evaluate(model, val_loader, criterion)
        recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far    = 100.0 * fp / (fp + tn) if (fp + tn) > 0 else 0.0

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        marker = " ✓" if va_acc > best_val_acc else ""

        print(f"  {epoch:>3}  {tr_loss:>7.4f}  {tr_acc:>5.1f}%  "
              f"{va_loss:>7.4f}  {va_acc:>5.1f}%  {recall:>6.1f}%  {far:>5.1f}%  "
              f"{mins}m{secs:02d}s{marker}")

        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve   = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "best_val_acc": best_val_acc}, teacher_ckpt)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT_TEACHER:
                print(f"\nEarly stopping — no improvement for {EARLY_STOP_PAT_TEACHER} epochs.")
                break

    print(f"\nTeacher best val accuracy: {best_val_acc:.1f}%")
    print(f"Saved → {teacher_ckpt}")


# ── Phase 2: Distill student ──────────────────────────────────────────────────
def train_student(data_root: str, num_epochs: int, resume: bool = False,
                  teacher_ckpt: Path = TEACHER_CKPT,
                  student_ckpt: Path = STUDENT_CKPT,
                  blank_weight: float = BLANK_WEIGHT,
                  lr: float = LR,
                  weight_decay: float = 0.00815,
                  grad_clip: float = GRAD_CLIP):
    print(f"\n{'='*60}")
    print("PHASE 2 — Student Distillation (BNNClassifier)")
    print(f"{'='*60}")
    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {num_epochs}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Alpha       : {ALPHA}  (KL weight; {1-ALPHA:.1f} on hard CE)")
    print(f"Dataset     : {data_root}")
    print(f"Teacher     : {teacher_ckpt}")
    print(f"Output      : {student_ckpt}\n")

    if not teacher_ckpt.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_ckpt}\n"
            "Run `distill.py teacher` first."
        )

    # Load teacher — frozen, eval mode only
    teacher = make_teacher().to(DEVICE)
    t_ckpt  = torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True)
    teacher.load_state_dict(t_ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded  (best val acc: {t_ckpt['best_val_acc']:.1f}%)\n")

    # Student
    student   = BNNClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    class_weights = torch.tensor([blank_weight, 1.0]).to(DEVICE)

    start_epoch  = 1
    best_val_acc = 0.0
    no_improve   = 0

    if resume and student_ckpt.exists():
        ckpt = torch.load(student_ckpt, map_location=DEVICE, weights_only=True)
        student.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val acc: {best_val_acc:.1f}%)\n")

    train_loader, val_loader = make_loaders(data_root)

    print(f"  {'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'Recall':>7}  {'FAR':>6}  {'Time':>6}")
    print(f"  {'─'*72}")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        student.train()
        optimizer.zero_grad()
        total_loss = total_correct = total_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}", unit="batch",
                    leave=False, disable=not sys.stdout.isatty())
        for step, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            student_logits = student(imgs)
            loss = distill_loss(student_logits, teacher_logits, labels, class_weights)
            (loss / ACCUM_STEPS).backward()

            last_batch = (step + 1) == len(train_loader)
            if (step + 1) % ACCUM_STEPS == 0 or last_batch:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()
                with torch.no_grad():
                    for m in student.modules():
                        if isinstance(m, BinarizeConv2d):
                            m.weight.data.clamp_(-1.0, 1.0)
                optimizer.zero_grad()

            total_loss    += loss.item() * len(labels)
            total_correct += (student_logits.argmax(1) == labels).sum().item()
            total_n       += len(labels)
            pbar.set_postfix(loss=f"{total_loss/total_n:.4f}", acc=f"{100.*total_correct/total_n:.1f}%")

        tr_loss = total_loss / total_n
        tr_acc  = 100.0 * total_correct / total_n

        # Reuse evaluate() from bnn_serengeti2 for val metrics
        val_criterion = nn.CrossEntropyLoss(weight=class_weights)
        val_loss, va_acc, tp, tn, fp, fn = evaluate(student, val_loader, val_criterion)
        recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far    = 100.0 * fp / (fp + tn) if (fp + tn) > 0 else 0.0

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
            }, student_ckpt)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"\nEarly stopping — val acc hasn't improved for {EARLY_STOP_PAT} epochs.")
                break

    print(f"\nDistilled student best val accuracy: {best_val_acc:.1f}%")
    print(f"Saved → {student_ckpt}")
    print(f"\nTo evaluate: python project/software_training/evaluate_bnn.py "
          f"--data-root project/data_20k --checkpoint {student_ckpt}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: ResNet-18 teacher → BNNClassifier student",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("phase", choices=["teacher", "student"],
                        help="'teacher' to fine-tune ResNet-50; 'student' to distill BNNClassifier")
    parser.add_argument("--data-root", default="project/data_20k", metavar="DIR")
    parser.add_argument("--epochs",    type=int, default=None,
                        help="Training epochs (default: 30 for teacher, 50 for student)")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume student from existing checkpoint (student phase only)")
    parser.add_argument("--teacher-checkpoint", default=None, metavar="PATH",
                        help="Teacher checkpoint path (default: bnn_teacher.pth)")
    parser.add_argument("--student-checkpoint", default=None, metavar="PATH",
                        help="Student checkpoint path (default: bnn_distilled.pth)")
    parser.add_argument("--blank-weight", type=float, default=BLANK_WEIGHT,
                        help=f"CrossEntropy weight for blank class (default: {BLANK_WEIGHT})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Student learning rate (default: {LR})")
    parser.add_argument("--weight-decay", type=float, default=0.00815,
                        help="Student AdamW weight decay (default: 0.00815)")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP,
                        help=f"Student gradient clip norm (default: {GRAD_CLIP})")
    args = parser.parse_args()

    _script_dir = Path(__file__).parent.parent
    teacher_ckpt = Path(args.teacher_checkpoint) if args.teacher_checkpoint else TEACHER_CKPT
    student_ckpt = Path(args.student_checkpoint) if args.student_checkpoint else STUDENT_CKPT
    if not teacher_ckpt.is_absolute():
        teacher_ckpt = _script_dir / teacher_ckpt
    if not student_ckpt.is_absolute():
        student_ckpt = _script_dir / student_ckpt

    if args.phase == "teacher":
        train_teacher(args.data_root, args.epochs or 30,
                      teacher_ckpt=teacher_ckpt, blank_weight=args.blank_weight)
    else:
        train_student(args.data_root, args.epochs or 50, resume=args.resume,
                      teacher_ckpt=teacher_ckpt, student_ckpt=student_ckpt,
                      blank_weight=args.blank_weight, lr=args.lr,
                      weight_decay=args.weight_decay, grad_clip=args.grad_clip)


if __name__ == "__main__":
    main()
