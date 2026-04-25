"""
label_ir_images.py
==================
Detect whether camera trap images are IR (night) or colour (day) by measuring
per-pixel channel divergence — independent of timestamp metadata.

Night IR images from camera traps are nearly greyscale: R ≈ G ≈ B because
the sensor records reflected infrared light with no colour filter active.
Daytime images have chromatic content (green vegetation, brown soil, blue sky).

Metric: mean max-channel-diff across all pixels.
  colourfulness = mean( max(|R-G|, |G-B|, |R-B|) )   per image
  score < COLOUR_THRESHOLD  → IR / night
  score ≥ COLOUR_THRESHOLD  → colour / day

When --metadata is supplied, Caltech-sourced images (blank_*/non_blank_*) are
cross-validated against their timestamp-derived labels so mislabelled images
can be quantified.

Usage:
  # Analyse all test images + cross-validate against Caltech timestamps
  python project/software_training/label_ir_images.py \\
      --data-root project/data_20k \\
      --metadata lila_metadata_cache.json.zip \\
      --out project/ser_tod_labels.csv

  # Serengeti images only (no metadata needed)
  python project/software_training/label_ir_images.py --data-root project/data_20k
"""

import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


COLOUR_THRESHOLD    = 10.0
CALTECH_IMAGE_BASE  = ("https://storage.googleapis.com/public-datasets-lila"
                        "/caltech-unzipped/cct_images")
DOWNLOAD_SEED_BLANK    = 42
DOWNLOAD_SEED_NONBLANK = 43


# ── Colourfulness metric ──────────────────────────────────────────────────────
def colourfulness(img_path: Path) -> float:
    """Return a scalar ≥ 0; near-zero means greyscale/IR."""
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return float(np.mean(np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)),
                                    np.abs(r - b))))


def ir_label(score: float) -> str:
    return "day" if score >= COLOUR_THRESHOLD else "night_ir"


# ── Timestamp map (mirrors evaluate_bnn / download_lila_dataset logic) ────────
def _build_date_map(metadata_path: str) -> dict[str, str]:
    path = Path(metadata_path)
    if not path.exists():
        return {}
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            with zf.open(name) as f:
                meta = json.load(f)
    else:
        with open(path) as f:
            meta = json.load(f)

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
        if not img.get("url"):
            if img.get("file_name"):
                img = {**img, "url": f"{CALTECH_IMAGE_BASE}/{img['file_name']}"}
            else:
                continue
        (blank_pool if cats <= empty_ids else animal_pool).append(img)

    random.seed(DOWNLOAD_SEED_BLANK)
    blank_sample = random.sample(blank_pool, min(10_000, len(blank_pool)))
    random.seed(DOWNLOAD_SEED_NONBLANK)
    animal_sample = random.sample(animal_pool, min(10_000, len(animal_pool)))

    date_map: dict[str, str] = {}
    for i, img in enumerate(blank_sample):
        date_map[f"blank_{i:05d}"] = img.get("date_captured", "")
    for i, img in enumerate(animal_sample):
        date_map[f"non_blank_{i:05d}"] = img.get("date_captured", "")
    return date_map


def _timestamp_tod(date_str: str) -> str:
    if not date_str:
        return "unknown"
    try:
        hour = int(date_str.split()[1].split(":")[0])
        return "day" if 7 <= hour <= 18 else "night"
    except (IndexError, ValueError):
        return "unknown"


# ── Main scan ─────────────────────────────────────────────────────────────────
def scan(data_root: Path, metadata_path: str | None = None,
         out_csv: Path | None = None, splits: tuple = ("train", "test")):

    date_map = _build_date_map(metadata_path) if metadata_path else {}
    if date_map:
        print(f"  Loaded timestamps for {len(date_map):,} Caltech images.\n")

    all_files = []
    for split in splits:
        for cls in ("blank", "non_blank"):
            d = data_root / split / cls
            if d.exists():
                all_files.extend((f, split, cls) for f in sorted(d.glob("*.jpg")))

    if not all_files:
        print("No images found.")
        return

    rows = []
    for f, split, cls in tqdm(all_files, desc="Classifying images", unit="img"):
        score     = colourfulness(f)
        ir_tod    = ir_label(score)
        stem      = f.stem
        ts_tod    = _timestamp_tod(date_map.get(stem, "")) if date_map else "unknown"
        source    = "serengeti" if stem.startswith("ser_") else "caltech"
        rows.append({
            "path":   str(f),
            "stem":   stem,
            "split":  split,
            "cls":    cls,
            "source": source,
            "score":  round(score, 2),
            "ir_tod": ir_tod,       # greyscale-based label  ← physical ground truth
            "ts_tod": ts_tod,       # timestamp-based label
        })

    # ── Overall IR summary ────────────────────────────────────────────────────
    total   = len(rows)
    n_day   = sum(1 for r in rows if r["ir_tod"] == "day")
    n_night = total - n_day
    print(f"\n{'═'*56}")
    print(f"  GREYSCALE CLASSIFIER RESULTS  ({total:,} images)")
    print(f"{'═'*56}")
    print(f"  Colour / day   : {n_day:,}  ({100*n_day/total:.1f}%)")
    print(f"  IR / night     : {n_night:,}  ({100*n_night/total:.1f}%)")
    print(f"  Threshold      : {COLOUR_THRESHOLD} (colourfulness score)\n")

    scores = sorted(r["score"] for r in rows)
    p10 = scores[int(len(scores) * 0.10)]
    p50 = scores[int(len(scores) * 0.50)]
    p90 = scores[int(len(scores) * 0.90)]
    print(f"  Score distribution  p10={p10:.1f}  median={p50:.1f}  p90={p90:.1f}\n")

    # ── Per-source breakdown ──────────────────────────────────────────────────
    for source in ("caltech", "serengeti"):
        src_rows = [r for r in rows if r["source"] == source]
        if not src_rows:
            continue
        sd = sum(1 for r in src_rows if r["ir_tod"] == "day")
        sn = len(src_rows) - sd
        print(f"  {source.capitalize():10s}  day={sd:,}  night_ir={sn:,}  total={len(src_rows):,}")
    print()

    # ── Cross-validation: timestamp vs greyscale (Caltech only) ──────────────
    caltech = [r for r in rows if r["source"] == "caltech" and r["ts_tod"] != "unknown"]
    if caltech:
        agree    = sum(1 for r in caltech if
                       (r["ts_tod"] == "day") == (r["ir_tod"] == "day"))
        disagree = len(caltech) - agree
        pct      = 100 * disagree / len(caltech)

        # Breakdown of disagreements
        ts_day_ir_night = sum(1 for r in caltech
                               if r["ts_tod"] == "day" and r["ir_tod"] == "night_ir")
        ts_night_ir_day = sum(1 for r in caltech
                               if r["ts_tod"] == "night" and r["ir_tod"] == "day")

        print(f"{'─'*56}")
        print(f"  TIMESTAMP vs GREYSCALE CROSS-VALIDATION  (Caltech, n={len(caltech):,})")
        print(f"{'─'*56}")
        print(f"  Agreement      : {agree:,}  ({100*agree/len(caltech):.1f}%)")
        print(f"  Disagreement   : {disagree:,}  ({pct:.1f}%)  ← likely mislabelled by timestamp")
        print(f"    Timestamp=day  but IR image : {ts_day_ir_night:,}")
        print(f"    Timestamp=night but colour  : {ts_night_ir_day:,}")
        print()
        print(f"  Implication: {pct:.1f}% of timestamped images have wrong day/night label.")
        print(f"  Greyscale classifier is the physical ground truth.\n")

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["path", "stem", "split", "cls", "source", "score", "ir_tod", "ts_tod"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Labels written → {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label camera trap images as IR/night or colour/day via greyscale detection")
    parser.add_argument("--data-root",  default="project/data_20k",  metavar="DIR")
    parser.add_argument("--metadata",   default=None, metavar="ZIP",
                        help="Caltech metadata zip for timestamp cross-validation "
                             "(e.g. lila_metadata_cache.json.zip)")
    parser.add_argument("--out",        default=None, metavar="CSV",
                        help="Output CSV path (e.g. project/ser_tod_labels.csv)")
    parser.add_argument("--threshold",  type=float, default=COLOUR_THRESHOLD,
                        help=f"Colourfulness threshold (default {COLOUR_THRESHOLD})")
    parser.add_argument("--splits",     nargs="+", default=["train", "test"],
                        help="Which splits to scan (default: train test)")
    args = parser.parse_args()

    COLOUR_THRESHOLD = args.threshold
    scan(Path(args.data_root), args.metadata, args.out, tuple(args.splits))
