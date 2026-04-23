"""
combine_datasets.py
===================
Merges the Serengeti dataset (1000×750) with the existing Caltech Camera
Traps dataset (already 224×224) into a single combined directory.

Serengeti images are resized to 224×224 in memory before saving.
Caltech images are symlinked (not copied) to save disk space.

Output structure:
  project/data_combined/
    train/blank/        ← 9,000 Caltech (symlink) + 1,413 Serengeti (resized)
    train/non_blank/    ← 9,000 Caltech (symlink) + 1,422 Serengeti (resized)
    test/blank/         ← 1,000 Caltech (symlink) +   282 Serengeti (resized)
    test/non_blank/     ← 1,000 Caltech (symlink) +   282 Serengeti (resized)

Usage:
  python combine_datasets.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from PIL import Image
from tqdm import tqdm

CALTECH_ROOT   = Path("project/data_20k")
SERENGETI_ROOT = Path("project/images/archive")
OUT_ROOT       = Path("project/data_combined")
IMG_SIZE       = 224
JPEG_QUALITY   = 90


def _symlink_caltech(split: str, cls: str):
    src_dir = CALTECH_ROOT / split / cls
    dst_dir = OUT_ROOT / split / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.jpg"))
    for f in files:
        link = dst_dir / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    return len(files)


def _copy_resize_serengeti(split: str, cls: str):
    src_dir = SERENGETI_ROOT / split / cls
    dst_dir = OUT_ROOT / split / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.jpg"))
    for f in tqdm(files, desc=f"  serengeti {split}/{cls}", unit="img"):
        dst = dst_dir / f"ser_{f.name}"
        if dst.exists():
            continue
        img = Image.open(f).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img.save(dst, "JPEG", quality=JPEG_QUALITY)
    return len(files)


def main():
    print("Building combined dataset …\n")

    totals = {}
    for split in ("train", "test"):
        for cls in ("blank", "non_blank"):
            n_cal = _symlink_caltech(split, cls)
            n_ser = _copy_resize_serengeti(split, cls)
            totals[f"{split}/{cls}"] = (n_cal, n_ser)

    print("\nDone. Directory: project/data_combined/\n")
    print(f"  {'Split':<20}  {'Caltech':>8}  {'Serengeti':>9}  {'Total':>7}")
    print(f"  {'─'*50}")
    for key, (n_cal, n_ser) in totals.items():
        print(f"  {key:<20}  {n_cal:>8,}  {n_ser:>9,}  {n_cal+n_ser:>7,}")

    total_imgs = sum(a + b for a, b in totals.values())
    print(f"\n  Grand total: {total_imgs:,} images")
    print("\nTo train on this dataset:")
    print("  python project/bnn_serengeti2.py train --data-root project/data_combined")


if __name__ == "__main__":
    main()
