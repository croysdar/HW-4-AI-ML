"""
download_lila_dataset.py
========================
Downloads 10,000 blank + 10,000 animal images from a LILA BC
COCO Camera Traps metadata JSON. Images are resized to 224×224
in memory before saving — no full-resolution files hit disk.

─── HOW TO GET THE METADATA JSON ────────────────────────────────────────────
1. Go to  https://lila.science/datasets/
2. Pick a dataset (e.g. "Caltech Camera Traps" or "Snapshot Serengeti S1")
3. Click the link labeled "COCO Camera Traps format metadata"
4. Copy that URL and pass it to --json-url, OR download it first and
   pass the local path to --json-file.

Large JSONs (Caltech is ~700 MB unzipped) are cached locally so you
only download them once.

─── USAGE ───────────────────────────────────────────────────────────────────
# Download the metadata JSON automatically and pull 10k+10k images:
python download_lila_dataset.py --json-url <METADATA_URL> --out-dir data/

# Use an already-downloaded metadata file:
python download_lila_dataset.py --json-file metadata.json --out-dir data/

# Adjust concurrency or sample size:
python download_lila_dataset.py --json-file metadata.json --out-dir data/ \\
    --n 10000 --workers 16

─── OUTPUT STRUCTURE ────────────────────────────────────────────────────────
<out-dir>/train/blank/        ← N empty/blank images (224×224 JPEG)
<out-dir>/train/non_blank/    ← N animal images      (224×224 JPEG)

Plug <out-dir> straight into bnn_serengeti2.py by setting DATA_ROOT.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import zipfile
import io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
JPEG_QUALITY = 90
TIMEOUT     = 20   # seconds per image request
MAX_RETRIES = 2


# ── JSON loading ──────────────────────────────────────────────────────────────
def _load_json(json_url: str | None, json_file: str | None) -> dict:
    """Download (if needed) and parse the COCO metadata JSON."""

    if json_file:
        path = Path(json_file)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        print(f"Loading metadata from {path} …")
        if str(path).endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".json"))
                with zf.open(name) as f:
                    return json.load(f)
        with open(path) as f:
            return json.load(f)

    # ── Download from URL ──────────────────────────────────────────────────
    # Cache to avoid re-downloading on re-runs
    cache_path = Path("lila_metadata_cache.json")
    if json_url.endswith(".zip"):
        cache_path = Path("lila_metadata_cache.json.zip")

    if not cache_path.exists():
        print(f"Downloading metadata JSON …\n  {json_url}")
        r = requests.get(json_url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(cache_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="metadata"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        print(f"Using cached metadata: {cache_path}")

    if str(cache_path).endswith(".zip"):
        print("Unzipping …")
        with zipfile.ZipFile(cache_path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            with zf.open(name) as f:
                return json.load(f)

    print("Parsing JSON …")
    with open(cache_path) as f:
        return json.load(f)


# Default image base URL for Caltech Camera Traps (confirmed working)
CALTECH_IMAGE_BASE = (
    "https://storage.googleapis.com/public-datasets-lila"
    "/caltech-unzipped/cct_images"
)


# ── Dataset parsing ───────────────────────────────────────────────────────────
def _split_image_pools(meta: dict, image_base_url: str = "") -> tuple[list[dict], list[dict]]:
    """
    Returns (blank_images, animal_images) by reading the COCO annotations.

    Empty/blank = any image whose ONLY annotations are in an "empty"-named
    category (case-insensitive). Animal = at least one non-empty annotation.
    Images with no annotations at all are skipped (unannotated).
    """
    # Build category id → name map
    cat_name = {c["id"]: c["name"].lower() for c in meta["categories"]}
    empty_ids = {cid for cid, name in cat_name.items() if "empty" in name}

    print(f"\nCategories found: {len(cat_name)}")
    print(f"  Empty category IDs : {empty_ids or 'none — check category names'}")
    if not empty_ids:
        # Fallback: show all category names so the user can diagnose
        print("  All categories:", sorted(cat_name.values()))
        raise ValueError(
            "No 'empty' category found. Check the category names above and "
            "set empty_ids manually in _split_image_pools()."
        )

    # Aggregate per-image category sets from annotations
    img_cats: dict[str, set] = defaultdict(set)
    for ann in meta["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])

    # Build id → image record map
    id_to_img = {img["id"]: img for img in meta["images"]}

    blank_pool, animal_pool = [], []
    for img_id, cats in img_cats.items():
        if img_id not in id_to_img:
            continue
        img = id_to_img[img_id]
        # Some datasets embed a url field; others only have file_name.
        if not img.get("url"):
            if image_base_url and img.get("file_name"):
                img = {**img, "url": f"{image_base_url}/{img['file_name']}"}
            else:
                continue      # no way to construct a URL — skip
        if cats <= empty_ids:  # all annotations are empty
            blank_pool.append(img)
        else:                  # at least one real animal
            animal_pool.append(img)

    print(f"\nAnnotated images: {len(img_cats):,}")
    print(f"  Blank pool : {len(blank_pool):,}")
    print(f"  Animal pool: {len(animal_pool):,}")
    return blank_pool, animal_pool


# ── Image download worker ─────────────────────────────────────────────────────
def _download_one(img_record: dict, dest_path: Path) -> str | None:
    """
    Download one image, resize to IMG_SIZE×IMG_SIZE, save as JPEG.
    Returns None on success, or an error string on failure.
    """
    if dest_path.exists():
        return None   # already downloaded — resume support

    url = img_record["url"]
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(dest_path, "JPEG", quality=JPEG_QUALITY)
            return None
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"{url}: {e}"
    return f"{url}: unknown error"


# ── Main download loop ────────────────────────────────────────────────────────
def _download_pool(
    pool: list[dict],
    dest_dir: Path,
    label: str,
    n: int,
    workers: int,
    seed: int,
) -> int:
    """Download n images from pool into dest_dir. Returns count of failures."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    sample = random.sample(pool, min(n, len(pool)))
    if len(sample) < n:
        print(f"  WARNING: only {len(sample):,} {label} images available (wanted {n:,})")

    failures, downloaded = [], 0
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as ex, tqdm(
        total=len(sample), desc=f"  {label}", unit="img"
    ) as bar:
        for i, img in enumerate(sample):
            # Stable filename: zero-padded index + .jpg
            fname = dest_dir / f"{label}_{i:05d}.jpg"
            fut   = ex.submit(_download_one, img, fname)
            futures[fut] = fname

        for fut in as_completed(futures):
            bar.update(1)
            err = fut.result()
            if err:
                failures.append(err)
            else:
                downloaded += 1

    if failures:
        log_path = dest_dir.parent / f"{label}_failures.txt"
        log_path.write_text("\n".join(failures))
        print(f"  {len(failures)} failures logged → {log_path}")

    return len(failures)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download LILA BC camera trap images (blank + animal, 224×224)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Provide the metadata JSON URL directly:
  python download_lila_dataset.py \\
      --json-url https://lila.science/public/... \\
      --out-dir data/

  # Use a locally-downloaded metadata file:
  python download_lila_dataset.py \\
      --json-file caltech_camera_traps.json \\
      --out-dir data/

Where to find the JSON URL:
  https://lila.science/datasets/
  → Click a dataset → copy the "COCO Camera Traps format" metadata link
        """,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json-url",  metavar="URL",  help="LILA BC COCO metadata JSON URL")
    src.add_argument("--json-file", metavar="PATH", help="Local metadata JSON (or .zip) path")

    parser.add_argument("--out-dir",  default="data", metavar="DIR",
                        help="Root output directory (default: data/)")
    parser.add_argument("--n",        type=int, default=10_000, metavar="N",
                        help="Images to download per class (default: 10000)")
    parser.add_argument("--workers",  type=int, default=8, metavar="N",
                        help="Parallel download threads (default: 8)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--image-base-url", default=CALTECH_IMAGE_BASE, metavar="URL",
                        help="Base URL prepended to file_name when images lack a url "
                             "field (default: Caltech GCS prefix)")

    args = parser.parse_args()

    # ── Parse metadata ──────────────────────────────────────────────────────
    meta = _load_json(args.json_url, args.json_file)
    blank_pool, animal_pool = _split_image_pools(meta, args.image_base_url)

    out = Path(args.out_dir)
    blank_dir    = out / "train" / "blank"
    nonblank_dir = out / "train" / "non_blank"

    print(f"\nDownloading {args.n:,} blank images  → {blank_dir}")
    print(f"Downloading {args.n:,} animal images → {nonblank_dir}")
    print(f"Workers: {args.workers}  |  Seed: {args.seed}\n")

    # ── Download ────────────────────────────────────────────────────────────
    blank_fails  = _download_pool(blank_pool,  blank_dir,    "blank",     args.n, args.workers, args.seed)
    animal_fails = _download_pool(animal_pool, nonblank_dir, "non_blank", args.n, args.workers, args.seed + 1)

    # ── Summary ─────────────────────────────────────────────────────────────
    blank_got  = len(list(blank_dir.glob("*.jpg")))
    animal_got = len(list(nonblank_dir.glob("*.jpg")))

    print(f"\n{'─'*50}")
    print(f"Done.")
    print(f"  blank/     : {blank_got:,} images  ({blank_fails} failures)")
    print(f"  non_blank/ : {animal_got:,} images  ({animal_fails} failures)")
    print(f"\nTo train on this dataset, update DATA_ROOT in bnn_serengeti2.py:")
    print(f"  DATA_ROOT = '{out.resolve()}'")


if __name__ == "__main__":
    main()
