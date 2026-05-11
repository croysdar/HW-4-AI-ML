"""
find_hard_night_blanks.py
=========================
Score all night blank images with the night BNN model, find the worst
false positives (highest p(animal)), and add them to data_sequences so
they are picked up as hard negatives during training and evaluation.

Each image is added as a single-frame sequence. The colourfulness filter
in _load_hard_blank_frames will correctly identify them as night sequences.

Usage:
  python project/software_training/find_hard_night_blanks.py \
      --checkpoint project/bnn_dualModel_2_night.pth \
      --data-root project/data_20k_night \
      --n 60 \
      --seq-dir project/data_sequences
"""

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from bnn_serengeti2 import BNNClassifier, _transform, DEVICE, _NONBLANK_IDX, _colourfulness, _COLOUR_THRESHOLD


def _load_model(ckpt_path: str) -> torch.nn.Module:
    m = BNNClassifier()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    return m.to(DEVICE).eval()


def score_blanks(data_root: Path, model: torch.nn.Module) -> list[tuple[float, Path]]:
    results = []
    for split in ["train", "test"]:
        blank_dir = data_root / split / "blank"
        if not blank_dir.exists():
            continue
        images = sorted(blank_dir.glob("*.jpg"))
        print(f"  Scoring {len(images):,} blank images from {split}/blank …")
        for img_path in tqdm(images, leave=False):
            img = _transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = torch.softmax(model(img), dim=1)[0, _NONBLANK_IDX].item()
            # Only include images the colourfulness metric confirms as night
            score = _colourfulness(img_path)
            if score < _COLOUR_THRESHOLD:
                results.append((p, img_path))
    results.sort(reverse=True)
    return results


def add_to_sequences(top: list[tuple[float, Path]], seq_dir: Path, n: int) -> None:
    index_path = seq_dir / "seq_index.json"
    index = json.loads(index_path.read_text())

    existing_sources = {
        e.get("source_image", "") for e in index
    }

    next_idx = max(e["seq_idx"] for e in index) + 1
    added = 0

    for p_animal, img_path in top:
        if added >= n:
            break
        if str(img_path) in existing_sources:
            continue

        seq_path = seq_dir / "blank" / f"seq_{next_idx:05d}"
        seq_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, seq_path / "frame_01.jpg")

        index.append({
            "seq_idx":      next_idx,
            "seq_id":       str(uuid.uuid4()),
            "label":        "blank",
            "n_frames":     1,
            "location":     "night_hard_negative",
            "date":         "2000-01-01 00:00:00",
            "source_image": str(img_path),
            "p_animal":     round(p_animal, 4),
            "frames":       ["frame_01"],
        })

        print(f"  seq_{next_idx:05d}  p(animal)={p_animal:.3f}  {img_path.name}")
        next_idx += 1
        added += 1

    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nAdded {added} night hard-blank sequences to {seq_dir}/seq_index.json")


def main():
    parser = argparse.ArgumentParser(description="Find and register hard night blank images")
    parser.add_argument("--checkpoint", default="project/bnn_dualModel_2_night.pth")
    parser.add_argument("--data-root",  default="project/data_20k_night", metavar="DIR")
    parser.add_argument("--n",          type=int, default=60,
                        help="Number of hard negatives to add (default: 60)")
    parser.add_argument("--seq-dir",    default="project/data_sequences", metavar="DIR")
    parser.add_argument("--threshold",  type=float, default=0.3,
                        help="Only add images with p(animal) above this (default: 0.3)")
    args = parser.parse_args()

    print(f"\nCheckpoint : {args.checkpoint}")
    print(f"Data root  : {args.data_root}")
    print(f"Target n   : {args.n}  (p(animal) > {args.threshold})\n")

    model    = _load_model(args.checkpoint)
    data_root = Path(args.data_root)
    seq_dir   = Path(args.seq_dir)

    all_scores = score_blanks(data_root, model)
    above_thresh = [(p, path) for p, path in all_scores if p >= args.threshold]

    print(f"\nFound {len(above_thresh):,} night blanks with p(animal) ≥ {args.threshold}")
    print(f"Adding top {min(args.n, len(above_thresh))} to {seq_dir}\n")

    add_to_sequences(above_thresh, seq_dir, args.n)


if __name__ == "__main__":
    main()
