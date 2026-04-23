"""
evaluate_bnn.py
===============
Deep-dive evaluation of the trained BNNClassifier on the validation set.

Outputs:
  1. Confusion matrix with TP / TN / FP / FN + derived metrics
  2. Day vs. Night accuracy breakdown (parsed from Caltech metadata)

Usage:
  python evaluate_bnn.py
  python evaluate_bnn.py --data-root project/data_20k --metadata lila_metadata_cache.json.zip
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

# Import model definition from the training script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from bnn_serengeti2 import (
    BNNClassifier, _transform, CHECKPOINT, DEVICE,
    _BLANK_IDX, _NONBLANK_IDX, _tta_probs,
)

# ── Metadata reconstruction ───────────────────────────────────────────────────
# download_lila_dataset.py renamed images to blank_NNNNN.jpg / non_blank_NNNNN.jpg.
# The index N maps back to sample[N] drawn with the same seeds as the download.
CALTECH_IMAGE_BASE = (
    "https://storage.googleapis.com/public-datasets-lila"
    "/caltech-unzipped/cct_images"
)
DOWNLOAD_SEED_BLANK    = 42
DOWNLOAD_SEED_NONBLANK = 43   # seed + 1 as used in download script


def _build_date_map(metadata_path: str) -> dict[str, str]:
    """
    Returns {filename_stem: date_captured_str} for all test images,
    reconstructing the download script's sampling order deterministically.
    """
    print("Loading metadata for day/night analysis …")
    path = Path(metadata_path)
    if not path.exists():
        print(f"  WARNING: metadata file not found at {metadata_path}")
        print("  Day/Night analysis will be skipped.")
        return {}

    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            with zf.open(name) as f:
                meta = json.load(f)
    else:
        with open(path) as f:
            meta = json.load(f)

    # Re-derive blank / animal pools (mirrors download_lila_dataset._split_image_pools)
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
        if cats <= empty_ids:
            blank_pool.append(img)
        else:
            animal_pool.append(img)

    # Re-sample with identical seeds → same order as download
    random.seed(DOWNLOAD_SEED_BLANK)
    blank_sample = random.sample(blank_pool, min(10_000, len(blank_pool)))

    random.seed(DOWNLOAD_SEED_NONBLANK)
    animal_sample = random.sample(animal_pool, min(10_000, len(animal_pool)))

    # Map filename stem → date_captured
    date_map: dict[str, str] = {}
    for i, img in enumerate(blank_sample):
        date_map[f"blank_{i:05d}"] = img.get("date_captured", "")
    for i, img in enumerate(animal_sample):
        date_map[f"non_blank_{i:05d}"] = img.get("date_captured", "")

    print(f"  Mapped {len(date_map):,} images to timestamps.\n")
    return date_map


# ── Day / Night helper ────────────────────────────────────────────────────────
def _time_of_day(date_str: str) -> str:
    """'day', 'night', or 'unknown' from a 'YYYY-MM-DD HH:MM:SS' string."""
    if not date_str:
        return "unknown"
    try:
        hour = int(date_str.split()[1].split(":")[0])
        return "day" if 7 <= hour <= 18 else "night"
    except (IndexError, ValueError):
        return "unknown"


# ── Inference ─────────────────────────────────────────────────────────────────
def run_evaluation(data_root: str, metadata_path: str, checkpoint: str, threshold: float = 0.5, use_tta: bool = False):
    test_dir = Path(data_root) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Load model
    model = BNNClassifier()
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(DEVICE)
    model.eval()

    # Dataset — use inference transform (no augmentation)
    dataset = datasets.ImageFolder(str(test_dir), transform=_transform)
    loader  = DataLoader(dataset, batch_size=64, num_workers=2,
                         persistent_workers=True, shuffle=False)

    # Date map for day/night (may be empty if metadata unavailable)
    date_map = _build_date_map(metadata_path)

    # Accumulators — overall and per time-of-day
    tp = tn = fp = fn = 0
    # Each tod bucket tracks its own TP/TN/FP/FN
    tod_counts = {
        "day":     {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "night":   {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "unknown": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
    }

    print(f"  TTA: {'enabled (4 views: orig, hflip, bright±0.15)' if use_tta else 'disabled'}\n")

    img_index = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs  = imgs.to(DEVICE)
            if use_tta:
                probs = _tta_probs(model, imgs).cpu()
            else:
                probs = torch.softmax(model(imgs), dim=1).cpu()
            # Apply threshold: predict ANIMAL only if p(animal) >= threshold
            preds = (probs[:, _NONBLANK_IDX] >= threshold).long()

            for pred, label in zip(preds.tolist(), labels.tolist()):
                actual_animal = (label == _NONBLANK_IDX)
                pred_animal   = (pred  == _NONBLANK_IDX)

                if actual_animal and pred_animal:
                    cell = "tp"; tp += 1
                elif not actual_animal and not pred_animal:
                    cell = "tn"; tn += 1
                elif not actual_animal and pred_animal:
                    cell = "fp"; fp += 1
                else:
                    cell = "fn"; fn += 1

                img_path = dataset.imgs[img_index][0]
                stem     = Path(img_path).stem
                tod      = _time_of_day(date_map.get(stem, ""))
                tod_counts[tod][cell] += 1

                img_index += 1

    # ── Print results ─────────────────────────────────────────────────────────
    total    = tp + tn + fp + fn
    accuracy = 100.0 * (tp + tn) / total
    precision = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    recall    = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    fpr       = 100.0 * fp / (fp + tn) if (fp + tn) else 0.0   # false alarm rate

    W = 52
    print("\n" + "═" * W)
    print(f"  CONFUSION MATRIX  (threshold = {threshold})")
    print("═" * W)
    print(f"  {'':20s}  {'Pred: ANIMAL':>13}  {'Pred: EMPTY':>11}")
    print(f"  {'─'*48}")
    print(f"  {'Actual: ANIMAL (non_blank)':26s}  TP = {tp:>5,}       FN = {fn:>5,}")
    print(f"  {'Actual: EMPTY  (blank)':26s}  FP = {fp:>5,}       TN = {tn:>5,}")
    print(f"  {'─'*48}")
    print(f"  Total images evaluated : {total:,}")
    print()
    print(f"  Overall Accuracy  : {accuracy:>6.1f}%")
    print(f"  Precision         : {precision:>6.1f}%  (ANIMAL alerts that were real)")
    print(f"  Recall            : {recall:>6.1f}%  (real animals we caught)")
    print(f"  False Alarm Rate  : {fpr:>6.1f}%  (empty images wrongly flagged)")

    if fp > fn:
        bias = f"biased toward FALSE ALARMS (FP {fp:,} > FN {fn:,})"
    elif fn > fp:
        bias = f"biased toward MISSED ANIMALS (FN {fn:,} > FP {fp:,})"
    else:
        bias = "balanced (FP == FN)"
    print(f"\n  Bias assessment   : {bias}")

    def _print_tod_block(label: str, c: dict):
        total_n = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        if total_n == 0:
            print(f"  {label}  —  (0 images)\n")
            return
        acc  = 100.0 * (c["tp"] + c["tn"]) / total_n
        rec  = 100.0 * c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        prec = 100.0 * c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        far  = 100.0 * c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0.0
        print(f"  {label}  ({total_n:,} images)")
        print(f"  {'─'*48}")
        print(f"  {'':20s}  {'Pred: ANIMAL':>13}  {'Pred: EMPTY':>11}")
        print(f"  {'Actual: ANIMAL':26s}  TP = {c['tp']:>5,}       FN = {c['fn']:>5,}")
        print(f"  {'Actual: EMPTY':26s}  FP = {c['fp']:>5,}       TN = {c['tn']:>5,}")
        print(f"  {'─'*48}")
        print(f"  Accuracy         : {acc:>5.1f}%")
        print(f"  Recall           : {rec:>5.1f}%  (animals caught)")
        print(f"  Precision        : {prec:>5.1f}%  (alerts that were real)")
        print(f"  False Alarm Rate : {far:>5.1f}%  (empty images wrongly flagged)\n")

    print("\n" + "═" * W)
    print("  DAY vs. NIGHT  —  FULL BREAKDOWN")
    print("═" * W + "\n")
    _print_tod_block("DAY   (07:00–18:00)", tod_counts["day"])
    _print_tod_block("NIGHT (19:00–06:00)", tod_counts["night"])
    if tod_counts["unknown"]["tp"] + tod_counts["unknown"]["tn"] + \
       tod_counts["unknown"]["fp"] + tod_counts["unknown"]["fn"] > 0:
        _print_tod_block("UNKNOWN (no timestamp)", tod_counts["unknown"])
    print("═" * W + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _SCRIPT_DIR = Path(__file__).parent

    parser = argparse.ArgumentParser(description="BNN deep-dive evaluation")
    parser.add_argument("--data-root", default=str(_SCRIPT_DIR / "data_20k"),
                        help="Dataset root containing test/ (default: project/data_20k)")
    parser.add_argument("--metadata",  default="lila_metadata_cache.json.zip",
                        help="Caltech metadata JSON or zip (default: lila_metadata_cache.json.zip)")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help=f"Model checkpoint .pth (default: {CHECKPOINT})")
    parser.add_argument("--threshold", type=float, default=None, metavar="T",
                        help="p(animal) threshold (default: sweep 0.5, 0.6, 0.7)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (4 views averaged)")
    args = parser.parse_args()

    if args.threshold is not None:
        run_evaluation(args.data_root, args.metadata, args.checkpoint, args.threshold, args.tta)
    else:
        # Sweep common thresholds so the tradeoff is visible at a glance
        for t in [0.5, 0.6, 0.7]:
            print(f"\n{'▶'*3}  THRESHOLD = {t}  {'◀'*3}")
            run_evaluation(args.data_root, args.metadata, args.checkpoint, t, args.tta)
