"""
evaluate_sequences.py
=====================
Runs the BNN classifier frame-by-frame through downloaded sequences and
evaluates the temporal filter in practice.

For each sequence, prints per-frame p(animal) and shows whether the
N-frame temporal filter would fire.

Usage:
  # Single model, 3-frame filter
  python project/software_training/evaluate_sequences.py \\
      --data-dir project/data_sequences \\
      --checkpoint project/bnn_serengeti2.pth

  # Ensemble, 3-frame filter, threshold 0.6
  python project/software_training/evaluate_sequences.py \\
      --data-dir project/data_sequences \\
      --checkpoint project/bnn_baseline_871pct.pth \\
      --ensemble  project/bnn_distilled_876pct.pth \\
      --threshold 0.6 --filter-n 3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from bnn_serengeti2 import BNNClassifier, _transform, DEVICE, _NONBLANK_IDX


def _load_model(ckpt_path: str) -> torch.nn.Module:
    model = BNNClassifier()
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(DEVICE).eval()
    return model


def _prob_animal(models: list, img_path: Path, threshold: float) -> tuple[float, bool]:
    img    = _transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = sum(torch.softmax(m(img), dim=1) for m in models) / len(models)
    p = probs[0, _NONBLANK_IDX].item()
    return p, p >= threshold


def _temporal_filter(detections: list[bool], n: int) -> list[bool]:
    """Return True at position i if detections[i-n+1 .. i] are all True."""
    triggered = []
    for i, d in enumerate(detections):
        window = detections[max(0, i - n + 1): i + 1]
        triggered.append(len(window) == n and all(window))
    return triggered


def _bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def run(data_dir: Path, checkpoints: list[str], threshold: float, filter_n: int):
    models = [_load_model(c) for c in checkpoints]
    label  = "ensemble" if len(models) > 1 else Path(checkpoints[0]).stem
    index  = json.loads((data_dir / "seq_index.json").read_text())

    # Sequence-level counters
    seq_tp = seq_tn = seq_fp = seq_fn = 0
    # Frame-level counters (before filter)
    frame_tp = frame_tn = frame_fp = frame_fn = 0

    print(f"\n{'═'*70}")
    print(f"  SEQUENCE TEMPORAL FILTER EVALUATION")
    print(f"  Model: {label}   Threshold: {threshold}   Filter: {filter_n}-frame")
    print(f"{'═'*70}\n")

    for entry in index:
        seq_dir   = data_dir / entry["label"] / f"seq_{entry['seq_idx']:05d}"
        true_label = entry["label"]   # "blank" or "animal"
        frames     = sorted(seq_dir.glob("frame_*.jpg"))

        if not frames:
            continue

        probs      = []
        detections = []
        for f in frames:
            p, det = _prob_animal(models, f, threshold)
            probs.append(p)
            detections.append(det)

            # Per-frame stats
            actual_animal = (true_label == "animal")
            if actual_animal and det:       frame_tp += 1
            elif not actual_animal and not det: frame_tn += 1
            elif not actual_animal and det:     frame_fp += 1
            else:                               frame_fn += 1

        triggered = _temporal_filter(detections, filter_n)
        fired      = any(triggered)

        # Sequence-level decision
        actual_animal = (true_label == "animal")
        if   actual_animal and fired:       seq_tp += 1; outcome = "✓ TP"
        elif not actual_animal and not fired: seq_tn += 1; outcome = "✓ TN"
        elif not actual_animal and fired:   seq_fp += 1; outcome = "✗ FP (false alarm)"
        else:                               seq_fn += 1; outcome = "✗ FN (missed)"

        # Print sequence summary
        loc  = entry.get("location", "?")
        date = entry.get("date", "")[:10]
        print(f"  [{true_label:6s}] loc={loc:>3}  {date}  →  {outcome}")
        for i, (f, p, det, trig) in enumerate(zip(frames, probs, detections, triggered)):
            det_sym  = "▲" if det  else "·"
            trig_sym = "🔔" if trig else "  "
            print(f"    frame {i+1}  {_bar(p)}  {p:.3f}  {det_sym} {trig_sym}")
        print()

    # ── Summary ────────────────────────────────────────────────────────────
    total_seqs   = seq_tp + seq_tn + seq_fp + seq_fn
    total_frames = frame_tp + frame_tn + frame_fp + frame_fn
    seq_acc   = 100 * (seq_tp + seq_tn) / total_seqs   if total_seqs   else 0
    frame_acc = 100 * (frame_tp + frame_tn) / total_frames if total_frames else 0

    seq_recall   = 100 * seq_tp   / (seq_tp + seq_fn)     if (seq_tp + seq_fn)     else 0
    frame_recall = 100 * frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) else 0
    seq_far      = 100 * seq_fp   / (seq_fp + seq_tn)     if (seq_fp + seq_tn)     else 0
    frame_far    = 100 * frame_fp / (frame_fp + frame_tn) if (frame_fp + frame_tn) else 0

    print(f"{'═'*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'':30s}  {'Per-frame':>10}  {'Sequence':>10}")
    print(f"  {'─'*54}")
    print(f"  {'Accuracy':30s}  {frame_acc:>9.1f}%  {seq_acc:>9.1f}%")
    print(f"  {'Recall (animal sequences)':30s}  {frame_recall:>9.1f}%  {seq_recall:>9.1f}%")
    print(f"  {'FAR (blank sequences)':30s}  {frame_far:>9.1f}%  {seq_far:>9.1f}%")
    print(f"  {'─'*54}")
    print(f"  Sequences  TP={seq_tp}  TN={seq_tn}  FP={seq_fp}  FN={seq_fn}  (n={total_seqs})")
    print(f"  Frames     TP={frame_tp}  TN={frame_tn}  FP={frame_fp}  FN={frame_fn}  (n={total_frames})")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate temporal filter on downloaded sequences")
    parser.add_argument("--data-dir",   default="project/data_sequences", metavar="DIR")
    parser.add_argument("--checkpoint", default="project/bnn_baseline_871pct.pth", metavar="CKPT")
    parser.add_argument("--ensemble",   default=None, metavar="CKPT",
                        help="Second checkpoint to average with primary")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--filter-n",   type=int,   default=3,
                        help="Consecutive frames required to trigger (default: 3)")
    args = parser.parse_args()

    checkpoints = [args.checkpoint]
    if args.ensemble:
        checkpoints.append(args.ensemble)

    run(Path(args.data_dir), checkpoints, args.threshold, args.filter_n)
