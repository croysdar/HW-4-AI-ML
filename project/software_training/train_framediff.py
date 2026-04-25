"""
train_framediff.py
==================
Full training pipeline for frame-difference wildlife detection.

Input to model:  α * |frame_t − frame_1| + (1−α) * frame_t
Architecture:    same BNNClassifier, trained from scratch on diff inputs.
Data:            3-frame Caltech Camera Traps sequences (not in data_20k).

Per-epoch concentration validation
───────────────────────────────────
After each epoch, Grad-CAM is run on a fixed held-out sample and the
Gini coefficient of each heatmap is measured:

  Gini = 0  →  heat perfectly uniform (texture / background matching)
  Gini = 1  →  heat on a single pixel (perfect localisation)

Reported columns:
  Gini-TP  mean Gini for animal frames the model called ANIMAL  (want high)
  Gini-FP  mean Gini for blank  frames the model called ANIMAL  (want low)

If frame-diff training is working the gap Gini-TP − Gini-FP should grow:
the model concentrates on moving animal pixels and ignores static backgrounds.

Setup (first run):
  # 1. Download 3-frame sequences (≈ 4 000 seqs, ~12 000 frames, ≈ 500 MB)
  python project/software_training/train_framediff.py --download-only \\
      --metadata lila_metadata_cache.json.zip

  # 2. Train
  python project/software_training/train_framediff.py \\
      --metadata lila_metadata_cache.json.zip

  # 3. Evaluate the diff model vs single-frame baseline
  python project/software_training/evaluate_bnn.py \\
      --checkpoint project/bnn_framediff.pth \\
      --data-root  project/data_20k          # ← single-frame test set still valid
"""

import io
import json
import os
import random
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent))

from bnn_serengeti2 import (
    ACCUM_STEPS, BATCH_SIZE, DEVICE, EARLY_STOP_PAT, EPOCHS,
    GRAD_CLIP, IMG_SIZE, LR, BNNClassifier, BinarizeConv2d,
    _transform, _NONBLANK_IDX,
)
from gradcam import _gradcam

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).parent
_PROJECT_DIR  = _SCRIPT_DIR.parent
SEQ_DIR       = _PROJECT_DIR / "data_3frame_seqs"
HB_SEQ_DIR    = _PROJECT_DIR / "data_sequences"   # 5-frame PIR-triggered blank seqs
CHECKPOINT    = str(_PROJECT_DIR / "bnn_framediff.pth")

DOWNLOAD_SEED_BLANK    = 42          # mirrors download_lila_dataset.py
DOWNLOAD_SEED_NONBLANK = 43
SEQ_SEED               = 314         # different from 5-frame eval (99) to get new sequences
MAX_SEQS_PER_CLASS     = 2000        # download up to 2k per class (~12k frames total)
VAL_RATIO              = 0.15        # fraction held out for validation + concentration eval
CONC_SAMPLE            = 40         # frames used for per-epoch Gini eval (fast)
ALPHA                  = 0.7        # diff blend weight (1.0 = pure diff, 0.0 = original)
CALTECH_IMAGE_BASE     = ("https://storage.googleapis.com/public-datasets-lila"
                           "/caltech-unzipped/cct_images")
IMG_TIMEOUT    = 20
MAX_RETRIES    = 2


# ── Gini coefficient ──────────────────────────────────────────────────────────
def _gini(cam: np.ndarray) -> float:
    """
    Spatial Gini coefficient of a Grad-CAM heatmap.
    0 = perfectly uniform (texture / global matching)
    1 = all heat on one pixel (perfect localisation)
    """
    x = cam.flatten().astype(np.float64)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


# ── Dataset ───────────────────────────────────────────────────────────────────
class FrameDiffDataset(Dataset):
    """
    Each item is one (frame_t, label) pair where frame_t has been blended
    with its within-sequence background:

        input = α * |frame_t − background| + (1−α) * frame_t

    background = frame_1 of the sequence (simulates a stored reference frame).

    For training we yield frames 2 and 3; for val we yield all 3.
    Blank label → 0,  Animal label → 1.
    """

    def __init__(self, seq_dir: Path, entries: list[dict],
                 alpha: float = ALPHA, augment: bool = True):
        self.seq_dir = seq_dir
        self.alpha   = alpha
        self.augment = augment
        self.items   = []   # (bg_path, frame_path, label_int)

        for entry in entries:
            label_int = 1 if entry["label"] == "animal" else 0
            seq_path  = seq_dir / entry["label"] / f"seq_{entry['seq_idx']:05d}"
            frames    = sorted(seq_path.glob("frame_*.jpg"))
            if len(frames) < 2:
                continue
            bg = frames[0]
            for f in frames[1:]:          # frames 2 and 3 during training
                self.items.append((bg, f, label_int))

    def __len__(self):
        return len(self.items)

    def _load_arr(self, path: Path) -> np.ndarray:
        return np.array(
            Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
            dtype=np.float32)

    def __getitem__(self, idx):
        bg_path, frame_path, label = self.items[idx]
        bg    = self._load_arr(bg_path)
        frame = self._load_arr(frame_path)

        residual = np.abs(frame - bg)
        blended  = np.clip(self.alpha * residual + (1 - self.alpha) * frame, 0, 255)

        pil = Image.fromarray(blended.astype(np.uint8))

        if self.augment:
            from torchvision import transforms
            aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.RandomGrayscale(p=0.15),
            ])
            pil = aug(pil)

        tensor = _transform(pil)
        return tensor, label


# ── Per-epoch concentration evaluation ───────────────────────────────────────
def _build_conc_sample(val_entries: list[dict], seq_dir: Path,
                       alpha: float, n: int = CONC_SAMPLE) -> list[tuple]:
    """
    Build a fixed list of (tensor, label_int, seq_label) tuples for Gini eval.
    Balanced between blank and animal, sampled once before training starts.
    """
    blank_entries  = [e for e in val_entries if e["label"] == "blank"]
    animal_entries = [e for e in val_entries if e["label"] == "animal"]
    half = n // 2

    def _pick_frame(entry):
        seq_path = seq_dir / entry["label"] / f"seq_{entry['seq_idx']:05d}"
        frames   = sorted(seq_path.glob("frame_*.jpg"))
        if len(frames) < 2:
            return None
        bg     = np.array(Image.open(frames[0]).convert("RGB").resize(
                           (IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        frame  = np.array(Image.open(frames[1]).convert("RGB").resize(
                           (IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        blend  = np.clip(alpha * np.abs(frame - bg) + (1 - alpha) * frame, 0, 255)
        tensor = _transform(Image.fromarray(blend.astype(np.uint8))).unsqueeze(0).to(DEVICE)
        label  = 1 if entry["label"] == "animal" else 0
        return tensor, label

    sample = []
    for entries, k in [(blank_entries, half), (animal_entries, half)]:
        random.shuffle(entries)
        for e in entries[:k]:
            item = _pick_frame(e)
            if item:
                sample.append(item)
    return sample


def _eval_concentration(model: nn.Module, sample: list[tuple],
                         layer: str = "bn3") -> tuple[float, float]:
    """
    Compute mean Gini for animal TPs and blank FPs.
    Returns (gini_tp, gini_fp).
    """
    model.eval()
    tp_ginis, fp_ginis = [], []

    for tensor, label in sample:
        with torch.no_grad():
            p_animal = torch.softmax(model(tensor), dim=1)[0, _NONBLANK_IDX].item()
        pred_animal = p_animal >= 0.5

        cam  = _gradcam(model, tensor, _NONBLANK_IDX, layer)
        gini = _gini(cam)

        if label == 1 and pred_animal:      # true positive
            tp_ginis.append(gini)
        elif label == 0 and pred_animal:    # false positive
            fp_ginis.append(gini)

    gini_tp = float(np.mean(tp_ginis)) if tp_ginis else float("nan")
    gini_fp = float(np.mean(fp_ginis)) if fp_ginis else float("nan")
    return gini_tp, gini_fp


# ── Download helpers (mirrors download_sequences.py) ─────────────────────────
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


def _find_3frame_sequences(meta: dict, exclude_ids: set) -> tuple[list, list]:
    cat_name  = {c["id"]: c["name"].lower() for c in meta["categories"]}
    empty_ids = {cid for cid, name in cat_name.items() if "empty" in name}
    img_cats: dict = defaultdict(set)
    for ann in meta["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])
    seq_to_imgs: dict = defaultdict(list)
    for img in meta["images"]:
        if img.get("seq_num_frames", 1) == 3:
            seq_to_imgs[img["seq_id"]].append(img)

    blank_seqs, animal_seqs = [], []
    for _, imgs in seq_to_imgs.items():
        if len(imgs) != 3:
            continue
        imgs = sorted(imgs, key=lambda x: x.get("frame_num", 0))
        if any(img["id"] in exclude_ids for img in imgs):
            continue
        frame_labels = []
        valid = True
        for img in imgs:
            cats = img_cats.get(img["id"])
            if not cats:
                valid = False; break
            if not img.get("url"):
                if img.get("file_name"):
                    img["url"] = f"{CALTECH_IMAGE_BASE}/{img['file_name']}"
                else:
                    valid = False; break
            frame_labels.append("blank" if cats <= empty_ids else "animal")
        if not valid or len(set(frame_labels)) != 1:
            continue
        (blank_seqs if frame_labels[0] == "blank" else animal_seqs).append(imgs)
    return blank_seqs, animal_seqs


def _download_frame(img: dict, dest: Path) -> str | None:
    if dest.exists():
        return None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(img["url"], timeout=IMG_TIMEOUT)
            r.raise_for_status()
            pil = Image.open(io.BytesIO(r.content)).convert("RGB")
            pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            pil.save(dest, "JPEG", quality=90)
            return None
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"{img['url']}: {e}"
    return None


def download_sequences(meta_path: str, out_dir: Path,
                       n: int = MAX_SEQS_PER_CLASS, workers: int = 16):
    print("Loading metadata …")
    meta = _load_meta(meta_path)
    print("Identifying training-set images to exclude …")
    exclude = _already_downloaded_ids(meta)
    print(f"  Excluding {len(exclude):,} images.\n")

    print("Searching for 3-frame sequences …")
    blank_seqs, animal_seqs = _find_3frame_sequences(meta, exclude)
    print(f"  Blank : {len(blank_seqs):,}   Animal: {len(animal_seqs):,}\n")

    index = []
    for label, seqs, seed_offset in [("blank", blank_seqs, 0),
                                      ("animal", animal_seqs, 1)]:
        random.seed(SEQ_SEED + seed_offset)
        chosen = random.sample(seqs, min(n, len(seqs)))
        if len(chosen) < n:
            print(f"  WARNING: only {len(chosen)} {label} sequences available")
        tasks = []
        for i, seq in enumerate(chosen):
            seq_path = out_dir / label / f"seq_{i:05d}"
            seq_path.mkdir(parents=True, exist_ok=True)
            for img in seq:
                fn  = img.get("frame_num", 0)
                dst = seq_path / f"frame_{fn:02d}.jpg"
                tasks.append((img, dst))
            index.append({
                "seq_idx": i, "seq_id": seq[0]["seq_id"],
                "label": label, "n_frames": len(seq),
                "location": seq[0].get("location", ""),
                "date": seq[0].get("date_captured", ""),
            })
        with ThreadPoolExecutor(max_workers=workers) as ex, \
             tqdm(total=len(tasks), desc=f"  {label}", unit="frame") as bar:
            futs = {ex.submit(_download_frame, img, dst): None for img, dst in tasks}
            for _ in as_completed(futs):
                bar.update(1)

    (out_dir / "seq_index.json").write_text(json.dumps(index, indent=2))
    total_frames = sum(e["n_frames"] for e in index)
    print(f"\nDownloaded {len(index)} sequences  ({total_frames} frames)")
    print(f"Index → {out_dir / 'seq_index.json'}")


# ── Per-frame HB evaluation (superseded by sequence-level below) ──────────────
# def _load_hb_diff_frames(seq_dir: Path, alpha: float) -> torch.Tensor | None:
#     """Load 5-frame PIR blank sequences as frame-diff tensors (frame_1 = background)."""
#     index_path = seq_dir / "seq_index.json"
#     if not index_path.exists():
#         return None
#     index = json.loads(index_path.read_text())
#     frames = []
#     for entry in index:
#         if entry["label"] != "blank":
#             continue
#         seq_path   = seq_dir / "blank" / f"seq_{entry['seq_idx']:05d}"
#         frame_files = sorted(seq_path.glob("frame_*.jpg"))
#         if len(frame_files) < 2:
#             continue
#         bg = np.array(Image.open(frame_files[0]).convert("RGB")
#                       .resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
#         for f in frame_files[1:]:
#             arr   = np.array(Image.open(f).convert("RGB")
#                              .resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
#             blend = np.clip(alpha * np.abs(arr - bg) + (1 - alpha) * arr, 0, 255)
#             frames.append(_transform(Image.fromarray(blend.astype(np.uint8))))
#     return torch.stack(frames) if frames else None
#
# def _hard_blank_far(model: nn.Module, frames: torch.Tensor,
#                     threshold: float = 0.5) -> float:
#     model.eval()
#     fp = tn = 0
#     with torch.no_grad():
#         for i in range(0, len(frames), 64):
#             probs = torch.softmax(model(frames[i:i + 64].to(DEVICE)), dim=1)
#             dets  = probs[:, _NONBLANK_IDX] >= threshold
#             fp   += int(dets.sum())
#             tn   += int((~dets).sum())
#     return 100.0 * fp / (fp + tn) if (fp + tn) else 0.0

# ── Sequence-level evaluation ─────────────────────────────────────────────────
def _diff_tensors_for_seq(seq_path: Path, alpha: float) -> list[torch.Tensor]:
    """Load one sequence as a list of diff tensors (frame_1 = background)."""
    frame_files = sorted(seq_path.glob("frame_*.jpg"))
    if len(frame_files) < 2:
        return []
    bg = np.array(Image.open(frame_files[0]).convert("RGB")
                  .resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    tensors = []
    for f in frame_files[1:]:
        arr   = np.array(Image.open(f).convert("RGB")
                         .resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        blend = np.clip(alpha * np.abs(arr - bg) + (1 - alpha) * arr, 0, 255)
        tensors.append(_transform(Image.fromarray(blend.astype(np.uint8))))
    return tensors


def _seq_predict(model: nn.Module, all_tensors: list[torch.Tensor],
                 seq_lengths: list[int], filter_n: int) -> list[int]:
    """Batch-infer all frames then vote per sequence. Returns list of 0/1 predictions."""
    if not all_tensors:
        return []
    batch = torch.stack(all_tensors).to(DEVICE)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(batch), 64):
            probs = torch.softmax(model(batch[i:i + 64]), dim=1)[:, _NONBLANK_IDX]
            preds.extend((probs >= 0.5).tolist())
    seq_preds, offset = [], 0
    for n in seq_lengths:
        seq_preds.append(1 if sum(preds[offset:offset + n]) >= filter_n else 0)
        offset += n
    return seq_preds


def evaluate_sequences(model: nn.Module, val_entries: list[dict],
                        seq_dir: Path, alpha: float,
                        filter_n: int = 1) -> tuple[float, float, float]:
    """
    Sequence-level evaluation with temporal filter.
    Returns (seq_acc, seq_recall, seq_far).
    """
    all_tensors, seq_lengths, labels = [], [], []
    for entry in val_entries:
        seq_path = seq_dir / entry["label"] / f"seq_{entry['seq_idx']:05d}"
        tensors  = _diff_tensors_for_seq(seq_path, alpha)
        if not tensors:
            continue
        all_tensors.extend(tensors)
        seq_lengths.append(len(tensors))
        labels.append(1 if entry["label"] == "animal" else 0)

    seq_preds = _seq_predict(model, all_tensors, seq_lengths, filter_n)
    tp = tn = fp = fn = 0
    for pred, label in zip(seq_preds, labels):
        if   label and pred:      tp += 1
        elif not label and pred:  fp += 1
        elif label and not pred:  fn += 1
        else:                     tn += 1
    n_animal = tp + fn
    n_blank  = fp + tn
    recall   = 100.0 * tp / n_animal if n_animal else 0.0
    far      = 100.0 * fp / n_blank  if n_blank  else 0.0
    seq_acc  = 100.0 * (tp + tn) / (tp + tn + fp + fn) if labels else 0.0
    return seq_acc, recall, far


def _hb_seq_far(model: nn.Module, seq_dir: Path, alpha: float,
                filter_n: int = 1) -> float:
    """Sequence-level FAR on PIR-triggered hard-blank sequences with temporal filter."""
    index_path = seq_dir / "seq_index.json"
    if not index_path.exists():
        return float("nan")
    index = json.loads(index_path.read_text())
    all_tensors, seq_lengths = [], []
    for entry in index:
        if entry["label"] != "blank":
            continue
        seq_path = seq_dir / "blank" / f"seq_{entry['seq_idx']:05d}"
        tensors  = _diff_tensors_for_seq(seq_path, alpha)
        if not tensors:
            continue
        all_tensors.extend(tensors)
        seq_lengths.append(len(tensors))
    if not seq_lengths:
        return float("nan")
    seq_preds = _seq_predict(model, all_tensors, seq_lengths, filter_n)
    fp = sum(seq_preds)
    tn = len(seq_preds) - fp
    return 100.0 * fp / (fp + tn)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module) -> tuple[float, float, int, int, int, int]:
    model.eval()
    total_loss = tp = tn = fp = fn = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * imgs.size(0)
            preds = (torch.softmax(logits, dim=1)[:, _NONBLANK_IDX] >= 0.5).long()
            for p, l in zip(preds.tolist(), labels.tolist()):
                a = l == _NONBLANK_IDX
                if   a and p:             tp += 1
                elif not a and not p:     tn += 1
                elif not a and p:         fp += 1
                else:                     fn += 1
    n   = tp + tn + fp + fn
    acc = 100.0 * (tp + tn) / n
    return total_loss / n, acc, tp, tn, fp, fn


# ── Training ──────────────────────────────────────────────────────────────────
def train(seq_dir: Path, epochs: int = EPOCHS, conc_layer: str = "bn3",
          warm_start: str | None = None, lr: float = LR, filter_n: int = 1):
    index_path = seq_dir / "seq_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No seq_index.json at {seq_dir}. Run with --download-only first.")

    all_entries = json.loads(index_path.read_text())
    random.seed(SEQ_SEED)
    random.shuffle(all_entries)
    split       = int(len(all_entries) * (1 - VAL_RATIO))
    train_entries = all_entries[:split]
    val_entries   = all_entries[split:]

    print(f"  Sequences — train: {len(train_entries)}  val: {len(val_entries)}")
    print(f"  Building datasets (alpha={ALPHA}) …")

    train_ds = FrameDiffDataset(seq_dir, train_entries, ALPHA, augment=True)
    val_ds   = FrameDiffDataset(seq_dir, val_entries,   ALPHA, augment=False)
    print(f"  Train items: {len(train_ds):,}   Val items: {len(val_ds):,}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    # Fixed concentration sample — built once, reused every epoch
    print(f"  Building concentration sample ({CONC_SAMPLE} frames) …")
    conc_sample = _build_conc_sample(val_entries, seq_dir, ALPHA, CONC_SAMPLE)
    print(f"  Ready ({len(conc_sample)} frames in sample)")

    hb_available = (HB_SEQ_DIR / "seq_index.json").exists()
    if hb_available:
        n_hb = sum(1 for e in json.loads((HB_SEQ_DIR / "seq_index.json").read_text())
                   if e["label"] == "blank")
        print(f"  Hard-blank sequences : {n_hb} (from {HB_SEQ_DIR.name}/)")
    print()

    model = BNNClassifier().to(DEVICE)
    if warm_start:
        ckpt = torch.load(warm_start, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"  Warm-start weights loaded from {warm_start}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00815)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.27, 1.0]).to(DEVICE))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_seq_acc = 0.0
    no_improve   = 0

    header = (f"\n{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  "
              f"{'VlLoss':>8}  {'SeqAcc':>7}  {'SeqRcl':>7}  {'SeqFAR':>7}  {'HB-FAR':>7}  "
              f"{'Gini-TP':>8}  {'Gini-FP':>8}  {'Time':>6}  {'LR':>8}")
    print(header)
    print(f"  (α={ALPHA}, filter_n={filter_n}, effective batch={BATCH_SIZE}×{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS})")
    print("─" * 112)

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = correct = n = 0
        optimizer.zero_grad()
        t0   = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}", unit="batch",
                    leave=False, disable=not sys.stdout.isatty())

        for step, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            (loss / ACCUM_STEPS).backward()
            run_loss += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            n        += imgs.size(0)
            pbar.set_postfix(loss=f"{run_loss/n:.4f}", acc=f"{100.*correct/n:.1f}%")
            last = (step + 1) == len(train_loader)
            if (step + 1) % ACCUM_STEPS == 0 or last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, BinarizeConv2d):
                            m.weight.data.clamp_(-1.0, 1.0)
                optimizer.zero_grad()

        t_loss = run_loss / n
        t_acc  = 100.0 * correct / n
        v_loss, _, _, _, _, _ = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        seq_acc, seq_rcl, seq_far = evaluate_sequences(
            model, val_entries, seq_dir, ALPHA, filter_n)
        hb_far   = _hb_seq_far(model, HB_SEQ_DIR, ALPHA, filter_n) if hb_available else float("nan")
        hb_far_s = f"{hb_far:>5.1f}%" if not np.isnan(hb_far) else "    n/a"

        # Concentration metric
        gini_tp, gini_fp = _eval_concentration(model, conc_sample, conc_layer)
        gini_tp_s = f"{gini_tp:.3f}" if not np.isnan(gini_tp) else "  n/a"
        gini_fp_s = f"{gini_fp:.3f}" if not np.isnan(gini_fp) else "  n/a"

        scheduler.step()

        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            no_improve   = 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_seq_acc": best_seq_acc,
                "alpha": ALPHA,
                "filter_n": filter_n,
            }, CHECKPOINT)
            marker = " ✓"
        else:
            no_improve += 1
            marker = ""

        epoch_time = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"
        lr_now     = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {t_loss:>8.4f}  {t_acc:>6.1f}%  "
              f"{v_loss:>8.4f}  {seq_acc:>6.1f}%  {seq_rcl:>6.1f}%  {seq_far:>6.1f}%  {hb_far_s:>7}  "
              f"{gini_tp_s:>8}  {gini_fp_s:>8}  {epoch_time:>6}  {lr_now:>8.2e}{marker}")

        if no_improve >= EARLY_STOP_PAT:
            print(f"\nEarly stopping — no improvement for {EARLY_STOP_PAT} epochs.")
            break

    print(f"\nBest seq accuracy : {best_seq_acc:.1f}%")
    print(f"Checkpoint        → {CHECKPOINT}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train BNN on frame-difference sequences")
    parser.add_argument("--metadata",      default="lila_metadata_cache.json.zip")
    parser.add_argument("--seq-dir",       default=str(SEQ_DIR))
    parser.add_argument("--download-only",   action="store_true",
                        help="Download sequences and exit (no training)")
    parser.add_argument("--force-download",  action="store_true",
                        help="Re-download even if seq-dir already exists")
    parser.add_argument("--max-seqs",      type=int, default=MAX_SEQS_PER_CLASS,
                        help=f"Sequences per class to download (default: {MAX_SEQS_PER_CLASS})")
    parser.add_argument("--workers",       type=int, default=16)
    parser.add_argument("--epochs",        type=int, default=EPOCHS)
    parser.add_argument("--alpha",         type=float, default=ALPHA,
                        help=f"Diff blend weight (default: {ALPHA})")
    parser.add_argument("--conc-layer",    default="bn3",
                        choices=["bn2", "bn3", "bn4"],
                        help="Grad-CAM layer for concentration metric (default: bn3)")
    parser.add_argument("--warm-start",   default=None, metavar="CKPT",
                        help="Load weights from this checkpoint before training "
                             "(e.g. project/bnn_serengeti2.pth)")
    parser.add_argument("--lr",           type=float, default=LR,
                        help=f"Initial learning rate (default: {LR})")
    parser.add_argument("--filter-n",     type=int, default=1,
                        help="Frames-per-sequence threshold for sequence-level alert (default: 1)")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)

    needs_download = (not seq_dir.exists()
                      or not (seq_dir / "seq_index.json").exists()
                      or args.force_download)
    if needs_download:
        if seq_dir.exists() and args.force_download:
            import shutil
            shutil.rmtree(seq_dir)
        print(f"Downloading sequences → {seq_dir}")
        seq_dir.mkdir(parents=True, exist_ok=True)
        download_sequences(args.metadata, seq_dir, args.max_seqs, args.workers)
        if args.download_only:
            sys.exit(0)
    elif args.download_only:
        print(f"Sequences already at {seq_dir} — nothing to download.")
        sys.exit(0)

    ALPHA = args.alpha
    print(f"\n{'='*60}")
    print(f"  Frame-Diff BNN Training")
    print(f"{'='*60}")
    print(f"  Seq dir  : {seq_dir}")
    print(f"  Alpha    : {ALPHA}")
    print(f"  Epochs   : {args.epochs}")
    if args.warm_start:
        print(f"  Warm-start: {args.warm_start}")
    print(f"  LR        : {args.lr:.2e}")
    print(f"  Checkpoint → {CHECKPOINT}\n")

    train(seq_dir, args.epochs, args.conc_layer, args.warm_start, args.lr, args.filter_n)
