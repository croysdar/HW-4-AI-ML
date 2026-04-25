"""
extract_bboxes.py
=================
Extracts bounding box annotations from the Caltech Camera Traps metadata
for every image we downloaded into data_20k, and saves them as a JSON file.

Not all images have bounding boxes — the Caltech dataset is primarily a
classification dataset. This script reports coverage and skips images without
boxes rather than failing.

Output format (project/bbox_annotations.json):
  {
    "blank_00000":     {"image_id": "...", "boxes": [{"category": "empty", "bbox": [x,y,w,h]}, ...]},
    "non_blank_00042": {"image_id": "...", "boxes": [{"category": "deer",  "bbox": [x,y,w,h]}]},
    "non_blank_00099": {"image_id": "...", "boxes": []},   ← annotated but no bbox
    ...
  }

bbox format: [x, y, width, height] in pixels of the original image
             (not normalised — divide by image width/height to get [0,1] range)

Usage:
  python project/software_training/extract_bboxes.py \\
      --metadata lila_metadata_cache.json.zip \\
      --out project/bbox_annotations.json
"""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

DOWNLOAD_SEED_BLANK    = 42
DOWNLOAD_SEED_NONBLANK = 43


def load_meta(path: str) -> dict:
    p = Path(path)
    if str(p).endswith(".zip"):
        with zipfile.ZipFile(p) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            with zf.open(name) as f:
                return json.load(f)
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Caltech bounding box annotations for downloaded images")
    parser.add_argument("--metadata",  default="lila_metadata_cache.json.zip", metavar="PATH")
    parser.add_argument("--bbox-file", default="caltech_bboxes_20200316.json",  metavar="PATH",
                        help="Separate LILA bbox JSON (caltech_bboxes_20200316.json)")
    parser.add_argument("--out",       default="project/bbox_annotations.json", metavar="PATH")
    args = parser.parse_args()

    print("Loading metadata …")
    meta = load_meta(args.metadata)

    # ── Build lookup structures ───────────────────────────────────────────────
    cat_name  = {c["id"]: c["name"] for c in meta["categories"]}
    empty_ids = {cid for cid, name in cat_name.items() if "empty" in name.lower()}
    id_to_img = {img["id"]: img for img in meta["images"]}

    # Load separate bbox file if provided — overrides base metadata annotations
    bbox_by_img: dict = defaultdict(list)
    bbox_cat_name: dict = {}
    bbox_img_dims: dict = {}   # image_id → (orig_width, orig_height)
    if args.bbox_file and Path(args.bbox_file).exists():
        print(f"Loading bbox file: {args.bbox_file} …")
        with open(args.bbox_file) as f:
            bbox_data = json.load(f)
        bbox_cat_name = {c["id"]: c["name"] for c in bbox_data.get("categories", [])}
        for img in bbox_data.get("images", []):
            if "width" in img and "height" in img:
                bbox_img_dims[img["id"]] = (img["width"], img["height"])
        for ann in bbox_data.get("annotations", []):
            if ann.get("bbox"):
                bbox_by_img[ann["image_id"]].append(ann)
        print(f"  {sum(len(v) for v in bbox_by_img.values()):,} bbox annotations "
              f"across {len(bbox_by_img):,} images")
    else:
        print("  No bbox file found — falling back to base metadata annotations (likely empty)")

    # Group base annotations by image_id for category lookup
    img_anns: dict = defaultdict(list)
    for ann in meta["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # ── Reconstruct download order (mirrors download_lila_dataset.py) ─────────
    img_cats: dict = defaultdict(set)
    for ann in meta["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])

    blank_pool, animal_pool = [], []
    for img_id, cats in img_cats.items():
        if img_id not in id_to_img:
            continue
        (blank_pool if cats <= empty_ids else animal_pool).append(id_to_img[img_id])

    random.seed(DOWNLOAD_SEED_BLANK)
    blank_sample = random.sample(blank_pool, min(10_000, len(blank_pool)))
    random.seed(DOWNLOAD_SEED_NONBLANK)
    animal_sample = random.sample(animal_pool, min(10_000, len(animal_pool)))

    # ── Extract bboxes for each downloaded image ──────────────────────────────
    output = {}
    n_with_bbox = 0

    def _process(sample, prefix):
        nonlocal n_with_bbox
        for i, img in enumerate(sample):
            stem  = f"{prefix}_{i:05d}"
            boxes = []
            # Prefer separate bbox file; fall back to base metadata
            bbox_anns = bbox_by_img.get(img["id"]) or []
            orig_w, orig_h = bbox_img_dims.get(img["id"], (None, None))
            if bbox_anns:
                for ann in bbox_anns:
                    cat = bbox_cat_name.get(ann["category_id"], "unknown")
                    boxes.append({
                        "category": cat,
                        "bbox": ann["bbox"],
                        "orig_width": orig_w,
                        "orig_height": orig_h,
                    })
                    n_with_bbox += 1
            else:
                for ann in img_anns.get(img["id"], []):
                    entry = {"category": cat_name.get(ann["category_id"], "unknown")}
                    entry["bbox"] = ann["bbox"] if ("bbox" in ann and ann["bbox"]) else None
                    if entry["bbox"]:
                        n_with_bbox += 1
                    boxes.append(entry)
            output[stem] = {
                "image_id":  img["id"],
                "file_name": img.get("file_name", ""),
                "boxes":     boxes,
            }

    print("Reconstructing blank sample …")
    _process(blank_sample, "blank")
    print("Reconstructing animal sample …")
    _process(animal_sample, "non_blank")

    # ── Report ────────────────────────────────────────────────────────────────
    total        = len(output)
    with_any_ann = sum(1 for v in output.values() if v["boxes"])
    with_bbox    = sum(1 for v in output.values()
                       if any(b["bbox"] for b in v["boxes"]))

    print(f"\n{'─'*50}")
    print(f"  Total images mapped   : {total:,}")
    print(f"  Have any annotation   : {with_any_ann:,}  ({100*with_any_ann/total:.1f}%)")
    print(f"  Have bounding box(es) : {with_bbox:,}  ({100*with_bbox/total:.1f}%)")
    print(f"  Total bbox entries    : {n_with_bbox:,}")
    print(f"{'─'*50}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved → {out_path}")
    print(f"\nNext: use bbox_annotations.json to overlay ground-truth boxes on")
    print(f"  Grad-CAM images, or as spatial supervision for RRR training loss.")


if __name__ == "__main__":
    main()
