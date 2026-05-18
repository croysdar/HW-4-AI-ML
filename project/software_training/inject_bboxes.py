#!/usr/bin/env python3
"""
inject_bboxes.py
================
Inject ground-truth bounding box SVG overlays (disappear on hover) into a
gallery HTML file.  Reads bbox coordinates from bbox_annotations.json and
scales them to 160×160 display size.

Usage:
  python project/software_training/inject_bboxes.py \
      --gallery project/missed_recalls_gallery.html \
      --bboxes  project/bbox_annotations.json
"""

import argparse
import json
import re
from pathlib import Path


def make_overlay(stem: str, bbox_data: dict) -> str:
    annot = bbox_data.get(stem, {})
    boxes = [b for b in annot.get("boxes", []) if b.get("bbox")]
    if not boxes:
        return ""

    parts = []
    for b in boxes:
        x, y, w, h = b["bbox"]
        orig_w = b.get("orig_width") or 1
        orig_h = b.get("orig_height") or 1
        sx = 160 / orig_w
        sy = 160 / orig_h
        rx, ry, rw, rh = x * sx, y * sy, w * sx, h * sy
        cat = b.get("category", "")
        parts.append(
            f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
            f'fill="none" stroke="#00ff88" stroke-width="1.5" rx="1"/>'
        )
        if cat:
            ty = max(ry - 2, 9)
            parts.append(
                f'<text x="{rx:.1f}" y="{ty:.1f}" '
                f'font-size="8" font-family="monospace" fill="#00ff88">{cat}</text>'
            )

    return (
        '<svg class="bbox-overlay" xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg>"
    )


EXTRA_CSS = """\
  .img-wrap { position:relative; display:inline-block; cursor:crosshair; }
  .img-wrap svg.bbox-overlay {
    position:absolute; top:0; left:0; width:160px; height:160px;
    pointer-events:none; transition:opacity 0.12s;
  }
  .img-wrap:hover svg.bbox-overlay { opacity:0; }
"""

# Match the <img> tag (base64 src) followed by <div class="name">filename</div>
CARD_RE = re.compile(
    r'(<img\s+src="data:image/jpeg;base64,[^"]+"[^>]*>)'
    r"(\s*\n\s*)"
    r'(<div class="name">([^<]+)</div>)'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="project/missed_recalls_gallery.html")
    ap.add_argument("--bboxes",  default="project/bbox_annotations.json")
    args = ap.parse_args()

    gallery_path = Path(args.gallery)
    bbox_data = json.loads(Path(args.bboxes).read_text())

    html = gallery_path.read_text(encoding="utf-8")

    # Inject CSS (only once, guard against re-running)
    if ".img-wrap" not in html:
        html = html.replace("</style>", EXTRA_CSS + "</style>", 1)

    n_with_box = 0
    n_no_box   = 0

    def replace_fn(m):
        nonlocal n_with_box, n_no_box
        img_tag   = m.group(1)
        whitespace = m.group(2)
        name_div  = m.group(3)
        filename  = m.group(4)
        stem = filename[:-4] if filename.endswith(".jpg") else filename
        overlay = make_overlay(stem, bbox_data)
        if overlay:
            n_with_box += 1
        else:
            n_no_box += 1
        return f'<div class="img-wrap">{img_tag}{overlay}</div>{whitespace}{name_div}'

    html = CARD_RE.sub(replace_fn, html)

    gallery_path.write_text(html, encoding="utf-8")
    print(f"Done → {gallery_path}  ({gallery_path.stat().st_size:,} bytes)")
    print(f"  {n_with_box} cards got bbox overlay, {n_no_box} had no annotation")


if __name__ == "__main__":
    main()
