"""
hard_neg_distill.py
===================
Targeted hard-negative distillation for the night BNN model.

The ResNet-50 night teacher achieves near-0% FAR on hard night blank sequences
while the BNN gets 74%. This script:

  1. Pre-computes teacher soft labels on all hard night blank frames.
  2. Trains the BNN student with two interleaved losses per epoch:
       a. Regular CE loss on normal training data  (preserves 86% accuracy)
       b. Weighted KL loss on hard blank frames     (transfers teacher knowledge)

The hard blank pass runs after each regular epoch with a configurable weight
multiplier so the teacher signal on hard blanks dominates relative to normal
training loss.

Usage:
  python project/software_training/hard_neg_distill.py \
      --teacher  project/bnn_teacher_night.pth \
      --student  project/bnn_dualModel_2_night.pth \
      --out      project/bnn_dualModel_2_night.pth \
      --data-root project/data_20k_night \
      --seq-dir   project/data_sequences \
      --epochs 20 --lr 5e-5 --hb-weight 8.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bnn_serengeti2 import (
    BNNClassifier, _transform, _train_transform, DEVICE,
    _NONBLANK_IDX, _colourfulness, _COLOUR_THRESHOLD,
    make_loaders, evaluate,
)
from distill import make_teacher

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TEMPERATURE = 3.0   # soften teacher distribution for KL loss


# ── Load hard blank frames + pre-compute teacher soft labels ──────────────────

def _load_hard_blank_night(seq_dir: Path, teacher: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load all night hard-blank frames, run teacher, return:
      frames  : [N, 3, 224, 224]  student inputs
      targets : [N, 2]            teacher soft labels (temperature-softened)
    """
    index = json.loads((seq_dir / "seq_index.json").read_text())
    frames = []
    for entry in index:
        if entry["label"] != "blank":
            continue
        seq_path = seq_dir / "blank" / f"seq_{entry['seq_idx']:05d}"
        sorted_frames = sorted(seq_path.glob("frame_*.jpg"))
        if not sorted_frames:
            continue
        score = _colourfulness(sorted_frames[0])
        if score >= _COLOUR_THRESHOLD:
            continue  # day sequence — skip
        for f in sorted_frames:
            frames.append(_transform(Image.open(f).convert("RGB")))

    if not frames:
        raise RuntimeError("No night hard-blank frames found in seq_dir")

    frame_tensor = torch.stack(frames)   # [N, 3, 224, 224]
    print(f"  Hard blank frames: {len(frames)} night")

    # Pre-compute teacher soft labels (fixed — teacher doesn't train)
    teacher.eval()
    soft_labels = []
    with torch.no_grad():
        for i in range(0, len(frames), 32):
            batch = frame_tensor[i:i+32].to(DEVICE)
            logits = teacher(batch)
            soft = F.softmax(logits / TEMPERATURE, dim=1).cpu()
            soft_labels.append(soft)
    soft_labels = torch.cat(soft_labels, dim=0)   # [N, 2]

    p_animal_mean = soft_labels[:, _NONBLANK_IDX].mean().item()
    print(f"  Teacher mean p(animal) on hard blanks: {p_animal_mean:.3f}  "
          f"(should be low — teacher correctly rejects these)")

    return frame_tensor, soft_labels


# ── Hard blank KL pass ────────────────────────────────────────────────────────

def _hard_blank_pass(student: nn.Module, optimizer: torch.optim.Optimizer,
                     frames: torch.Tensor, targets: torch.Tensor,
                     hb_weight: float, grad_clip: float, n_passes: int = 2) -> float:
    """
    Run n_passes through all hard blank frames with KL distillation loss.
    Returns mean loss.
    """
    student.train()
    total_loss = 0.0
    steps = 0
    for _ in range(n_passes):
        perm = torch.randperm(len(frames))
        for i in range(0, len(frames), 32):
            idx = perm[i:i+32]
            batch   = frames[idx].to(DEVICE)
            soft_tgt = targets[idx].to(DEVICE)

            optimizer.zero_grad()
            logits = student(batch)
            log_soft_student = F.log_softmax(logits / TEMPERATURE, dim=1)
            kl = F.kl_div(log_soft_student, soft_tgt, reduction="batchmean") * (TEMPERATURE ** 2)
            loss = hb_weight * kl
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    return total_loss / max(steps, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hard-negative distillation for night BNN")
    parser.add_argument("--teacher",   default="project/bnn_teacher_night.pth")
    parser.add_argument("--student",   default="project/bnn_dualModel_2_night.pth")
    parser.add_argument("--out",       default="project/bnn_dualModel_2_night.pth")
    parser.add_argument("--data-root", default="project/data_20k_night")
    parser.add_argument("--seq-dir",   default="project/data_sequences")
    parser.add_argument("--epochs",    type=int,   default=20)
    parser.add_argument("--lr",        type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.005564)
    parser.add_argument("--grad-clip", type=float, default=0.756)
    parser.add_argument("--hb-weight", type=float, default=8.0,
                        help="KL loss multiplier for hard blank pass (default: 8.0)")
    parser.add_argument("--hb-passes", type=int, default=2,
                        help="Passes through hard blanks per epoch (default: 2)")
    parser.add_argument("--best-acc",  type=float, default=0.0,
                        help="Min val acc to save checkpoint (default: from checkpoint)")
    args = parser.parse_args()

    print(f"\nDevice  : {DEVICE}")
    print(f"Teacher : {args.teacher}")
    print(f"Student : {args.student}")
    print(f"HB weight: {args.hb_weight}x  |  HB passes/epoch: {args.hb_passes}")
    print(f"LR: {args.lr}  |  Epochs: {args.epochs}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    teacher = make_teacher().to(DEVICE)
    t_ckpt  = torch.load(args.teacher, map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(t_ckpt["model"] if "model" in t_ckpt else t_ckpt, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = BNNClassifier().to(DEVICE)
    s_ckpt  = torch.load(args.student, map_location=DEVICE, weights_only=False)
    student.load_state_dict(s_ckpt["model"] if "model" in s_ckpt else s_ckpt, strict=False)

    best_val_acc = args.best_acc if args.best_acc > 0 else s_ckpt.get("best_val_acc", 0.0)
    print(f"Must beat {best_val_acc:.2f}% to save → {args.out}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(args.data_root)
    criterion = nn.CrossEntropyLoss()

    hb_frames, hb_targets = _load_hard_blank_night(Path(args.seq_dir), teacher)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(student.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f" {'Ep':>3}  {'TrnLoss':>8}  {'TrnAcc':>7}  {'HBLoss':>7}  "
          f"{'ValAcc':>7}  {'Recall':>7}  {'FAR':>6}  {'HB-FAR':>7}   LR")
    print("-" * 95)

    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        # ── Regular training pass ─────────────────────────────────────────
        student.train()
        trn_loss = trn_correct = trn_total = 0
        for imgs, labels, _ in tqdm(train_loader, leave=False, desc=f"Ep{epoch} train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(student(imgs), labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            trn_loss    += loss.item() * len(labels)
            trn_correct += (student(imgs).argmax(1) == labels).sum().item()
            trn_total   += len(labels)

        trn_loss /= trn_total
        trn_acc   = 100.0 * trn_correct / trn_total

        # ── Hard blank distillation pass ──────────────────────────────────
        hb_loss = _hard_blank_pass(student, optimizer, hb_frames, hb_targets,
                                    args.hb_weight, args.grad_clip, args.hb_passes)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Validation ────────────────────────────────────────────────────
        val_loss, val_acc, tp, tn, fp, fn = evaluate(student, val_loader, criterion)
        recall = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
        far    = 100.0 * fp / (fp + tn) if (fp + tn) else 0.0

        # HB-FAR on hard blank frames
        student.eval()
        hb_fp = hb_tn = 0
        with torch.no_grad():
            for i in range(0, len(hb_frames), 64):
                p = torch.softmax(student(hb_frames[i:i+64].to(DEVICE)), dim=1)
                dets = p[:, _NONBLANK_IDX] >= 0.5
                hb_fp += int(dets.sum())
                hb_tn += int((~dets).sum())
        hb_far = 100.0 * hb_fp / (hb_fp + hb_tn) if (hb_fp + hb_tn) else 0.0

        mark = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": student.state_dict(), "epoch": epoch,
                        "best_val_acc": best_val_acc}, args.out)
            mark = " ✓"
            no_improve = 0
        else:
            no_improve += 1

        print(f" {epoch:3d}  {trn_loss:8.4f}  {trn_acc:6.1f}%  {hb_loss:7.4f}  "
              f"{val_acc:6.1f}%  {recall:6.1f}%  {far:5.1f}%  {hb_far:6.1f}%  "
              f" {current_lr:.2e}{mark}")

        if no_improve >= 15:
            print(f"\nEarly stopping — no improvement for 15 epochs.")
            break

    print(f"\nBest val accuracy : {best_val_acc:.2f}%")
    print(f"Checkpoint saved  → {args.out}")


if __name__ == "__main__":
    main()
