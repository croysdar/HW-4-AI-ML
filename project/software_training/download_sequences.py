"""
download_sequences.py
=====================
Downloads complete N-frame camera trap sequences from Caltech metadata,
keeping them separate from the training set for temporal filter evaluation.

Output structure:
  project/data_sequences/
    blank/
      seq_00000/  frame_01.jpg  frame_02.jpg  … frame_05.jpg
      seq_00001/  …
    animal/
      seq_00000/  …
    seq_index.json   ← metadata for each downloaded sequence

Usage:
  python project/software_training/download_sequences.py \\
      --metadata lila_metadata_cache.json.zip \\
      --out-dir project/data_sequences \\
      --n 50 --frames 5 --workers 16
"""

import argparse
import io
import json
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

CALTECH_IMAGE_BASE = (
    "https://storage.googleapis.com/public-datasets-lila"
    "/caltech-unzipped/cct_images"
)
IMG_SIZE     = 224
JPEG_QUALITY = 90
TIMEOUT      = 20
MAX_RETRIES  = 2
# Seeds used by download_lila_dataset.py — needed to exclude those images
DOWNLOAD_SEED_BLANK    = 42
DOWNLOAD_SEED_NONBLANK = 43


def _load_meta(path: str) -> dict:
    p = Path(path)
    if str(p).endswith(".zip"):
        with zipfile.ZipFile(p) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            with zf.open(name) as f:
                return json.load(f)
    with open(p) as f:
        return json.load(f)


def _already_downloaded_ids(meta: dict) -> set:
    """Reconstruct the image IDs already in data_20k to avoid overlap."""
    cat_name  = {c["id"]: c["name"].lower() for c in meta["categories"]}
    empty_ids = {cid for cid, name in cat_name.items() if "empty" in name}

    img_cats: dict = defaultdict(set)
    for ann in meta["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])

    id_to_img = {img["id"]: img for img in meta["images"]}
    blank_pool, animal_pool = [], []
    for img_id, cats in img_cats.items():
        if img_id not in id_to_img:
            continue
        img = id_to_img[img_id]
        (blank_pool if cats <= empty_ids else animal_pool).append(img)

    random.seed(DOWNLOAD_SEED_BLANK)
    blank_sample = random.sample(blank_pool, min(10_000, len(blank_pool)))
    random.seed(DOWNLOAD_SEED_NONBLANK)
    animal_sample = random.sample(animal_pool, min(10_000, len(animal_pool)))

    return {img["id"] for img in blank_sample + animal_sample}


def _find_sequences(meta: dict, min_frames: int,
                    exclude_ids: set) -> tuple[list, list]:
    """
    Returns (blank_seqs, animal_seqs) where each entry is a list of image
    dicts sorted by frame_num. Only includes sequences where:
      - every frame has exactly min_frames frames in the sequence
      - no frame is in exclude_ids (not in our training set)
      - all frames are consistently blank OR consistently animal
    """
    cat_name  = {c["id"]: c["name"].lower() for c in meta["categories"]}
    empty_ids = {cid for cid, name in cat_name.items() if "empty" in name}

    img_cats: dict = defaultdict(set)
    for ann in meta["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])

    # Group images by seq_id
    seq_to_imgs: dict = defaultdict(list)
    for img in meta["images"]:
        if img.get("seq_num_frames", 1) >= min_frames:
            seq_to_imgs[img["seq_id"]].append(img)

    blank_seqs, animal_seqs = [], []

    for seq_id, imgs in seq_to_imgs.items():
        # Must have exactly min_frames frames
        if len(imgs) != min_frames:
            continue
        # Sort by frame_num
        imgs = sorted(imgs, key=lambda x: x.get("frame_num", 0))
        # No frame already in training set
        if any(img["id"] in exclude_ids for img in imgs):
            continue
        # All frames must be annotated and have a URL
        frame_labels = []
        valid = True
        for img in imgs:
            cats = img_cats.get(img["id"])
            if not cats:
                valid = False
                break
            if not img.get("url"):
                if img.get("file_name"):
                    img["url"] = f"{CALTECH_IMAGE_BASE}/{img['file_name']}"
                else:
                    valid = False
                    break
            frame_labels.append("blank" if cats <= empty_ids else "animal")
        if not valid:
            continue
        # All frames must agree on blank vs animal
        if len(set(frame_labels)) != 1:
            continue
        if frame_labels[0] == "blank":
            blank_seqs.append(imgs)
        else:
            animal_seqs.append(imgs)

    return blank_seqs, animal_seqs


def _download_frame(img: dict, dest: Path) -> str | None:
    if dest.exists():
        return None
    url = img["url"]
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            pil = Image.open(io.BytesIO(r.content)).convert("RGB")
            pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            pil.save(dest, "JPEG", quality=JPEG_QUALITY)
            return None
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"{url}: {e}"
    return f"{url}: unknown error"


def _download_sequences(seqs: list, out_dir: Path, label: str,
                         n: int, workers: int, seed: int) -> list[dict]:
    """Download n sequences, return list of metadata dicts for seq_index.json."""
    random.seed(seed)
    chosen = random.sample(seqs, min(n, len(seqs)))
    if len(chosen) < n:
        print(f"  WARNING: only {len(chosen)} {label} sequences available (wanted {n})")

    index = []
    tasks = []   # (img_dict, dest_path, seq_idx, frame_num)
    for i, seq in enumerate(chosen):
        seq_dir = out_dir / label / f"seq_{i:05d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for img in seq:
            frame_num = img.get("frame_num", 0)
            dest = seq_dir / f"frame_{frame_num:02d}.jpg"
            tasks.append((img, dest))
        index.append({
            "seq_idx":   i,
            "seq_id":    seq[0]["seq_id"],
            "label":     label,
            "n_frames":  len(seq),
            "location":  seq[0].get("location", ""),
            "date":      seq[0].get("date_captured", ""),
            "frames":    [img["id"] for img in seq],
        })

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex, \
         tqdm(total=len(tasks), desc=f"  {label} sequences", unit="frame") as bar:
        futs = {ex.submit(_download_frame, img, dest): (img, dest)
                for img, dest in tasks}
        for fut in as_completed(futs):
            bar.update(1)
            err = fut.result()
            if err:
                failures.append(err)

    if failures:
        print(f"  {len(failures)} frame failures for {label}")
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Download complete N-frame Caltech sequences for temporal filter evaluation")
    parser.add_argument("--metadata",  default="lila_metadata_cache.json.zip", metavar="PATH")
    parser.add_argument("--out-dir",   default="project/data_sequences",       metavar="DIR")
    parser.add_argument("--n",         type=int, default=50,
                        help="Sequences per class to download (default: 50)")
    parser.add_argument("--frames",    type=int, default=5,
                        help="Exact frame count per sequence (default: 5)")
    parser.add_argument("--workers",   type=int, default=16, metavar="N")
    parser.add_argument("--seed",      type=int, default=99)
    args = parser.parse_args()

    print("Loading metadata …")
    meta = _load_meta(args.metadata)

    print("Identifying already-downloaded images …")
    exclude = _already_downloaded_ids(meta)
    print(f"  Excluding {len(exclude):,} images already in training set.\n")

    print(f"Searching for {args.frames}-frame sequences …")
    blank_seqs, animal_seqs = _find_sequences(meta, args.frames, exclude)
    print(f"  Blank sequences  : {len(blank_seqs):,}")
    print(f"  Animal sequences : {len(animal_seqs):,}\n")

    out = Path(args.out_dir)
    index = []

    print(f"Downloading {args.n} blank sequences …")
    index += _download_sequences(blank_seqs, out, "blank",  args.n, args.workers, args.seed)

    print(f"Downloading {args.n} animal sequences …")
    index += _download_sequences(animal_seqs, out, "animal", args.n, args.workers, args.seed + 1)

    (out / "seq_index.json").write_text(json.dumps(index, indent=2))

    total_frames = sum(e["n_frames"] for e in index)
    print(f"\n{'─'*50}")
    print(f"Done.")
    print(f"  {len(index)} sequences  ({total_frames} frames total)")
    print(f"  Index → {out / 'seq_index.json'}")
    print(f"\nNext: run evaluate_sequences.py to test the temporal filter.")


if __name__ == "__main__":
    main()
