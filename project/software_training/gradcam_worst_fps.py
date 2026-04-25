"""
gradcam_worst_fps.py
====================
Finds the blank test images with the highest p(animal) score (worst false
positives), runs Grad-CAM on each, and writes them all to an output directory
with a gallery.html for easy visual inspection.

Usage:
  python project/software_training/gradcam_worst_fps.py \\
      --checkpoint project/bnn_distilled_876pct.pth \\
      --n 20 \\
      --out-dir project/gradcam_worst_fps

  # Ensemble
  python project/software_training/gradcam_worst_fps.py \\
      --checkpoint project/bnn_baseline_871pct.pth \\
      --ensemble   project/bnn_distilled_876pct.pth \\
      --n 20
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from bnn_serengeti2 import BNNClassifier, _transform, DEVICE, _NONBLANK_IDX, CHECKPOINT, DATA_ROOT
from gradcam import run as gradcam_run, rebuild_gallery


def _load_model(ckpt_path: str) -> torch.nn.Module:
    m = BNNClassifier()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return m.to(DEVICE).eval()


def find_worst_fps(data_root: str, checkpoints: list[str],
                   n: int, threshold: float) -> list[tuple[float, Path]]:
    """Return [(p_animal, path), ...] for the n blank test images with highest p(animal)."""
    blank_dir = Path(data_root) / "test" / "blank"
    if not blank_dir.exists():
        raise FileNotFoundError(f"Blank test dir not found: {blank_dir}")

    models = [_load_model(c) for c in checkpoints]

    results = []
    images  = sorted(blank_dir.glob("*.jpg"))
    print(f"  Scoring {len(images):,} blank test images …")

    for img_path in images:
        img = _transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = sum(torch.softmax(m(img), dim=1) for m in models) / len(models)
        p = probs[0, _NONBLANK_IDX].item()
        results.append((p, img_path))

    results.sort(reverse=True)
    worst = results[:n]

    print(f"\n  Top {n} false positives (blank images most confidently called ANIMAL):")
    print(f"  {'File':<25}  p(animal)")
    print(f"  {'─'*36}")
    for p, path in worst:
        marker = " ← above threshold" if p >= threshold else ""
        print(f"  {path.name:<25}  {p:.3f}{marker}")
    return worst


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM on worst false positives from the blank test set")
    parser.add_argument("--checkpoint", default=CHECKPOINT, metavar="CKPT")
    parser.add_argument("--ensemble",   default=None, metavar="CKPT")
    parser.add_argument("--data-root",  default=DATA_ROOT, metavar="DIR")
    parser.add_argument("--n",          type=int, default=20,
                        help="Number of worst FPs to visualise (default: 20)")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--layer",      default="bn3",
                        choices=["bn2", "bn3", "bn4"])
    parser.add_argument("--out-dir",    default="project/gradcam_worst_fps", metavar="DIR")
    args = parser.parse_args()

    checkpoints = [args.checkpoint]
    if args.ensemble:
        checkpoints.append(args.ensemble)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint(s): {[Path(c).name for c in checkpoints]}")
    print(f"Looking in   : {args.data_root}/test/blank\n")

    worst = find_worst_fps(args.data_root, checkpoints, args.n, args.threshold)

    print(f"\nRunning Grad-CAM on top {args.n} …\n")
    for rank, (p, img_path) in enumerate(worst, 1):
        out = str(out_dir / f"rank{rank:02d}_{img_path.stem}_gradcam.jpg")
        gradcam_run(str(img_path), checkpoints, out, args.threshold, args.layer)

    rebuild_gallery(out_dir)
    print(f"\nDone. Open {out_dir}/gallery.html to inspect.")


if __name__ == "__main__":
    main()
