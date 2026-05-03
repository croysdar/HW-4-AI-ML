"""
occlusion_sensitivity.py
========================
Model-agnostic spatial diagnostic for the BNN wildlife classifier.

For each image, slides a gray patch over the input in a grid, re-runs the
model at each position, and records the drop in p(animal). Regions where
occlusion causes a large drop are genuinely important to the decision —
no gradient assumptions required.

  bright = important (covering this region kills the animal score)
  dark   = model doesn't care about this region

Ground-truth bounding boxes are overlaid in green when available, so you
can directly compare where the model looks vs where the animal actually is.

Usage:
  # Run on 20 random annotated images (default)
  python project/software_training/occlusion_sensitivity.py

  # Run on specific images
  python project/software_training/occlusion_sensitivity.py \\
      --images project/data_20k/train/non_blank/non_blank_00042.jpg

  # Finer grid (slower but more precise)
  python project/software_training/occlusion_sensitivity.py --patch 16 --stride 8
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bnn_serengeti2 import (BNNClassifier, _transform, _NONBLANK_IDX, DEVICE, CHECKPOINT,
                            _BORDER_PX)
from gradcam import _to_b64

_PROJECT_DIR = Path(__file__).parent.parent
_BBOX_PATH   = _PROJECT_DIR / "bbox_annotations.json"


# ── Core occlusion map ────────────────────────────────────────────────────────
def occlusion_map(model: torch.nn.Module, img_tensor: torch.Tensor,
                  patch: int = 32, stride: int = 16,
                  fill: float = 0.0) -> tuple[np.ndarray, float]:
    """
    Returns (heatmap [H,W], base_p_animal).
    heatmap[y,x] = mean drop in p(animal) when that pixel is inside the patch.
    Positive = important, negative = occluding this region raises the score.
    """
    _, C, H, W = img_tensor.shape

    with torch.no_grad():
        base_p = torch.softmax(model(img_tensor), dim=1)[0, _NONBLANK_IDX].item()

    B = _BORDER_PX
    masked_batch, positions = [], []
    for y in range(B, H - B - patch + 1, stride):
        for x in range(B, W - B - patch + 1, stride):
            m = img_tensor.clone()
            m[0, :, y:y + patch, x:x + patch] = fill
            masked_batch.append(m)
            positions.append((y, x))

    occ_scores = []
    batch = torch.cat(masked_batch, dim=0)
    with torch.no_grad():
        for i in range(0, len(batch), 64):
            probs = torch.softmax(model(batch[i:i + 64].to(DEVICE)), dim=1)
            occ_scores.extend(probs[:, _NONBLANK_IDX].tolist())

    heatmap = np.zeros((H, W), dtype=np.float32)
    count   = np.zeros((H, W), dtype=np.float32)
    for (y, x), score in zip(positions, occ_scores):
        drop = base_p - score
        heatmap[y:y + patch, x:x + patch] += drop
        count  [y:y + patch, x:x + patch] += 1.0

    count = np.maximum(count, 1)
    return heatmap / count, base_p


# ── Visualisation helpers ─────────────────────────────────────────────────────
def _heatmap_to_pil(heatmap: np.ndarray, orig: Image.Image,
                    alpha: float = 0.55) -> Image.Image:
    """Overlay occlusion heatmap on original image using jet colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = heatmap.copy()
    # Clip negative values (occluding raised score — not interesting here)
    h = np.clip(h, 0, None)
    if h.max() > 0:
        h = h / h.max()

    cmap   = plt.get_cmap("jet")
    colored = (cmap(h)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.fromarray(colored).resize(orig.size, Image.BILINEAR)
    return Image.blend(orig, overlay, alpha)


def _apply_mask_overlay(pil: Image.Image) -> Image.Image:
    """Draw black border over the regions _MaskBanner zeros out."""
    img  = pil.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    B = _BORDER_PX
    draw.rectangle([0, 0, W, B], fill="black")       # top
    draw.rectangle([0, H - B, W, H], fill="black")   # bottom
    draw.rectangle([0, 0, B, H], fill="black")        # left
    draw.rectangle([W - B, 0, W, H], fill="black")    # right
    return img


def bbox_alignment(heatmap: np.ndarray, boxes: list, size: int = 224) -> float | None:
    """Fraction of positive heatmap energy that falls inside the ground-truth bbox union.
    Returns None if no valid bboxes are present."""
    mask = np.zeros((size, size), dtype=np.float32)
    for b in boxes:
        if not b.get("bbox"):
            continue
        x, y, w, h = b["bbox"]
        ow = b.get("orig_width")  or size
        oh = b.get("orig_height") or size
        sx, sy = size / ow, size / oh
        x1 = max(0, int(x * sx))
        y1 = max(0, int(y * sy))
        x2 = min(size, int((x + w) * sx) + 1)
        y2 = min(size, int((y + h) * sy) + 1)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    if mask.sum() == 0:
        return None
    positive = np.clip(heatmap, 0, None)
    total = positive.sum()
    if total == 0:
        return None
    return float((positive * mask).sum() / total)


def _draw_boxes(pil: Image.Image, boxes: list) -> Image.Image:
    img  = pil.copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        if not b.get("bbox"):
            continue
        x, y, w, h = b["bbox"]
        ow = b.get("orig_width")  or 224
        oh = b.get("orig_height") or 224
        sx, sy = 224 / ow, 224 / oh
        draw.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy],
                       outline="lime", width=2)
    return img


# ── HTML output ───────────────────────────────────────────────────────────────
def _card(stem: str, orig: Image.Image, occ_pil: Image.Image,
          boxes: list, p_animal: float, alignment: float | None,
          max_drop: float = 0.0) -> str:
    orig_box = _draw_boxes(orig, boxes)
    occ_box  = _draw_boxes(occ_pil, boxes)
    p_colour = "#e03c3c" if p_animal >= 0.5 else "#888"
    cats     = ", ".join(set(b.get("category", "?") for b in boxes if b.get("bbox")))
    if alignment is not None:
        a_colour = "#3cb87a" if alignment >= 0.5 else ("#f0a500" if alignment >= 0.25 else "#e03c3c")
        align_str = f'<span style="color:{a_colour}"> align={alignment:.1%}</span>'
    else:
        align_str = '<span style="color:#888"> align=n/a</span>'
    drop_colour = "#3cb87a" if max_drop >= 0.05 else ("#f0a500" if max_drop >= 0.01 else "#888")
    return f"""
<div class="card">
  <div class="label">{stem}
    <span style="color:{p_colour}"> p(animal)={p_animal:.3f}</span>
    {align_str}
    <span style="color:{drop_colour}"> max_drop={max_drop:.4f}</span>
    <small> {cats}</small>
  </div>
  <div class="row">
    <div><div class="sub">Original + bbox</div>
         <img src="data:image/jpeg;base64,{_to_b64(orig_box)}"></div>
    <div><div class="sub">Occlusion sensitivity + bbox</div>
         <img src="data:image/jpeg;base64,{_to_b64(occ_box)}"></div>
  </div>
</div>"""


def _build_html(cards: list[str], checkpoint: str, patch: int, stride: int,
                mean_alignment: float | None = None) -> str:
    align_str = f"mean align={mean_alignment:.1%}" if mean_alignment is not None else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body{{background:#111;color:#ddd;font-family:monospace;padding:16px}}
h2{{font-size:15px;margin-bottom:4px}}
.meta{{font-size:11px;color:#888;margin-bottom:16px}}
.card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;
       padding:10px;margin-bottom:14px}}
.label{{font-size:13px;margin-bottom:6px}}
.sub{{font-size:11px;color:#888;margin-bottom:3px}}
.row{{display:flex;gap:10px}}
.row img{{width:224px;height:224px;display:block;border-radius:3px}}
</style></head>
<body>
<h2>Occlusion Sensitivity</h2>
<p class="meta">checkpoint: {checkpoint} &nbsp;|&nbsp;
  patch={patch}px  stride={stride}px &nbsp;|&nbsp;
  bright = important  dark = ignored &nbsp;|&nbsp;
  green box = ground-truth annotation &nbsp;|&nbsp;
  align = heatmap energy inside bbox &nbsp;|&nbsp; {align_str}</p>
{''.join(cards)}
</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def run(image_paths: list[Path], checkpoint: str, out_path: Path,
        patch: int, stride: int, bbox_path: Path = _BBOX_PATH,
        seed: int = 7, n_random: int = 20):

    print(f"Loading checkpoint: {checkpoint}")
    ckpt  = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model = BNNClassifier().to(DEVICE).eval()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    bboxes = json.loads(bbox_path.read_text()) if bbox_path.exists() else {}

    # Default: same 20 random annotated images as bbox_check.html
    if not image_paths:
        candidates = [
            (stem, info) for stem, info in bboxes.items()
            if stem.startswith("non_blank")
            and any(b.get("bbox") for b in info["boxes"])
        ]
        random.seed(seed)
        chosen = random.sample(candidates, min(n_random, len(candidates)))
        image_paths = []
        for stem, _ in chosen:
            for split in ("train", "test"):
                p = _PROJECT_DIR / "data_20k" / split / "non_blank" / f"{stem}.jpg"
                if p.exists():
                    image_paths.append(p)
                    break

    cards      = []
    alignments = []
    for img_path in image_paths:
        stem  = img_path.stem
        orig  = Image.open(img_path).convert("RGB").resize((224, 224))
        tensor = _transform(orig).unsqueeze(0).to(DEVICE)

        print(f"  {stem} …", end=" ", flush=True)
        heatmap, p_animal = occlusion_map(model, tensor, patch=patch, stride=stride)

        boxes     = bboxes.get(stem, {}).get("boxes", [])
        alignment = bbox_alignment(heatmap, boxes)
        max_drop  = float(np.clip(heatmap, 0, None).max())
        if alignment is not None:
            alignments.append(alignment)
        align_out = f"{alignment:.1%}" if alignment is not None else "n/a"
        print(f"p={p_animal:.3f}  align={align_out}  max_drop={max_drop:.4f}")

        masked_orig = _apply_mask_overlay(orig)
        occ_pil = _heatmap_to_pil(heatmap, masked_orig)
        cards.append(_card(stem, masked_orig, occ_pil, boxes, p_animal, alignment, max_drop))

    if alignments:
        print(f"\nMean bbox alignment : {sum(alignments)/len(alignments):.1%}  "
              f"(n={len(alignments)}, ≥50%: {sum(a>=0.5 for a in alignments)}/{len(alignments)})")

    mean_align = sum(alignments) / len(alignments) if alignments else None
    html = _build_html(cards, checkpoint, patch, stride, mean_align)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occlusion sensitivity for BNN classifier")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--images",     nargs="*", default=[],
                        help="Image paths to analyse (default: 20 random annotated non-blanks)")
    parser.add_argument("--patch",      type=int, default=32,
                        help="Occlusion patch size in pixels (default: 32)")
    parser.add_argument("--stride",     type=int, default=16,
                        help="Patch stride in pixels (default: 16)")
    parser.add_argument("--out",        default="project/occlusion_outputs/occlusion.html")
    args = parser.parse_args()

    run(
        image_paths=[Path(p) for p in args.images],
        checkpoint=args.checkpoint,
        out_path=Path(args.out),
        patch=args.patch,
        stride=args.stride,
    )
