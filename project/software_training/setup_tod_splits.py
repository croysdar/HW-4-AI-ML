"""
setup_tod_splits.py
===================
Creates data_20k_day/ and data_20k_night/ directories with symlinks to images
from data_20k/ based on ir_tod labels in ser_tod_labels.csv.

Directory layout produced:
  data_20k_day/
    train/blank/      -> symlinks to day blank train images
    train/non_blank/  -> symlinks to day non_blank train images
    test/blank/       -> symlinks to day blank test images
    test/non_blank/   -> symlinks to day non_blank test images
  data_20k_night/     -> same structure for night_ir images

Usage:
  python software_training/setup_tod_splits.py
"""

import csv
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_CSV_PATH = _PROJECT_DIR / "ser_tod_labels.csv"
_DATA_ROOT = _PROJECT_DIR / "data_20k"

SPLITS = {
    "day":   _PROJECT_DIR / "data_20k_day",
    "night": _PROJECT_DIR / "data_20k_night",
}


def main():
    for out_dir in SPLITS.values():
        for split in ("train", "test"):
            for cls in ("blank", "non_blank"):
                (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {k: {"train": 0, "test": 0, "missing": 0} for k in SPLITS}

    with open(_CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ir_tod = row["ir_tod"]
            tod_key = "day" if ir_tod == "day" else "night"

            src = (_PROJECT_DIR.parent / row["path"]).resolve()
            if not src.exists():
                counts[tod_key]["missing"] += 1
                continue

            split = row["split"]   # "train" or "test"
            cls   = row["cls"]     # "blank" or "non_blank"
            dst   = SPLITS[tod_key] / split / cls / src.name

            if not dst.exists():
                dst.symlink_to(src)
            counts[tod_key][split] += 1

    for tod, out_dir in SPLITS.items():
        c = counts[tod]
        print(f"\n{tod.upper()} → {out_dir}")
        print(f"  train: {c['train']:,}  test: {c['test']:,}  missing: {c['missing']:,}")
        for split in ("train", "test"):
            for cls in ("blank", "non_blank"):
                n = len(list((out_dir / split / cls).iterdir()))
                print(f"    {split}/{cls}: {n:,}")


if __name__ == "__main__":
    main()
