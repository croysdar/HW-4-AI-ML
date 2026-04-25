"""
gradcam_worst_fps.py
====================
Finds extreme test images and runs Grad-CAM on each:
  --mode fp  : worst false positives (blank images most confidently called ANIMAL)
  --mode tp  : best true positives  (animal images most confidently called ANIMAL)

Gallery output lets you compare what the model actually "sees" in each case.

Usage:
  # Worst false positives (default)
  python project/software_training/gradcam_worst_fps.py \\
      --checkpoint project/bnn_distilled_876pct.pth --n 16

  # Best true positives
  python project/software_training/gradcam_worst_fps.py \\
      --checkpoint project/bnn_distilled_876pct.pth --n 16 --mode tp \\
      --out-dir project/gradcam_best_tps
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


def find_extreme(data_root: str, checkpoints: list[str], n: int,
                 threshold: float, mode: str) -> list[tuple[float, Path]]:
    """
    mode='fp': worst false positives — blank images scored highest p(animal)
    mode='tp': best true positives  — animal images scored highest p(animal)
    Returns [(p_animal, path), ...] sorted by p(animal) descending.
    """
    cls_dir = Path(data_root) / "test" / ("blank" if mode == "fp" else "non_blank")
    if not cls_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {cls_dir}")

    models = [_load_model(c) for c in checkpoints]
    label  = "blank (FP)" if mode == "fp" else "animal (TP)"
    images = sorted(cls_dir.glob("*.jpg"))
    print(f"  Scoring {len(images):,} {label} test images …")

    results = []
    for img_path in images:
        img = _transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = sum(torch.softmax(m(img), dim=1) for m in models) / len(models)
        results.append((probs[0, _NONBLANK_IDX].item(), img_path))

    results.sort(reverse=True)
    top = results[:n]

    title = f"Top {n} {'worst false positives' if mode == 'fp' else 'best true positives'}"
    print(f"\n  {title}:")
    print(f"  {'File':<35}  p(animal)")
    print(f"  {'─'*46}")
    for p, path in top:
        marker = " ← above threshold" if p >= threshold else ""
        print(f"  {path.name:<35}  {p:.3f}{marker}")
    return top


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM on worst false positives from the blank test set")
    parser.add_argument("--checkpoint", default=CHECKPOINT, metavar="CKPT")
    parser.add_argument("--ensemble",   default=None, metavar="CKPT")
    parser.add_argument("--data-root",  default=DATA_ROOT, metavar="DIR")
    parser.add_argument("--mode",       default="fp", choices=["fp", "tp"],
                        help="fp=worst false positives, tp=best true positives (default: fp)")
    parser.add_argument("--n",          type=int, default=20,
                        help="Number of images to visualise (default: 20)")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--layer",      default="bn3", choices=["bn2", "bn3", "bn4"])
    parser.add_argument("--out-dir",    default=None, metavar="DIR",
                        help="Output directory (default: project/gradcam_worst_fps or gradcam_best_tps)")
    args = parser.parse_args()

    checkpoints = [args.checkpoint]
    if args.ensemble:
        checkpoints.append(args.ensemble)

    default_dir = "project/gradcam_best_tps" if args.mode == "tp" else "project/gradcam_worst_fps"
    out_dir = Path(args.out_dir or default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint(s): {[Path(c).name for c in checkpoints]}")
    print(f"Mode         : {'best true positives (animal)' if args.mode == 'tp' else 'worst false positives (blank)'}\n")

    top = find_extreme(args.data_root, checkpoints, args.n, args.threshold, args.mode)

    print(f"\nRunning Grad-CAM on top {args.n} …\n")
    for rank, (p, img_path) in enumerate(top, 1):
        out = str(out_dir / f"rank{rank:02d}_{img_path.stem}_gradcam.jpg")
        gradcam_run(str(img_path), checkpoints, out, args.threshold, args.layer)

    rebuild_gallery(out_dir)
    print(f"\nDone. Open {out_dir}/gallery.html to inspect.")


if __name__ == "__main__":
    main()
