"""
hard_blank_gallery.py
=====================
Generate an HTML gallery of all hard night blank sequences,
showing p(animal) from both the ResNet-50 teacher and BNN student.

Usage:
  python project/software_training/hard_blank_gallery.py \
      --teacher project/bnn_teacher_night.pth \
      --student project/bnn_dualModel_2_night.pth \
      --seq-dir project/data_sequences \
      --out     project/hard_blank_gallery.html
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bnn_serengeti2 import (
    BNNClassifier, _transform, DEVICE,
    _NONBLANK_IDX, _colourfulness, _COLOUR_THRESHOLD,
)
from distill import make_teacher

def _score(model, img_tensor):
    with torch.no_grad():
        return torch.softmax(model(img_tensor.unsqueeze(0).to(DEVICE)), dim=1)[0, _NONBLANK_IDX].item()


def _b64(img_path: Path, size: int = 224) -> str:
    img = Image.open(img_path).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher",  default="project/bnn_teacher_night.pth")
    parser.add_argument("--student",  default="project/bnn_dualModel_2_night.pth")
    parser.add_argument("--seq-dir",  default="project/data_sequences")
    parser.add_argument("--out",      default="project/hard_blank_gallery.html")
    args = parser.parse_args()

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading teacher …")
    teacher = make_teacher().to(DEVICE)
    t_ckpt  = torch.load(args.teacher, map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(t_ckpt["model"] if "model" in t_ckpt else t_ckpt, strict=False)
    teacher.eval()

    print("Loading student …")
    student = BNNClassifier().to(DEVICE)
    s_ckpt  = torch.load(args.student, map_location=DEVICE, weights_only=False)
    student.load_state_dict(s_ckpt["model"] if "model" in s_ckpt else s_ckpt, strict=False)
    student.eval()
    student_acc = s_ckpt.get("best_val_acc", 0.0)

    # ── Collect night hard blank sequences ────────────────────────────────────
    seq_dir = Path(args.seq_dir)
    index   = json.loads((seq_dir / "seq_index.json").read_text())

    entries = []
    for entry in index:
        if entry["label"] != "blank":
            continue
        seq_path = seq_dir / "blank" / f"seq_{entry['seq_idx']:05d}"
        frames   = sorted(seq_path.glob("frame_*.jpg"))
        if not frames:
            continue
        score = _colourfulness(frames[0])
        if score >= _COLOUR_THRESHOLD:
            continue
        entries.append((entry, frames))

    print(f"Found {len(entries)} night hard-blank sequences ({sum(len(f) for _,f in entries)} frames)")

    # ── Score all frames ──────────────────────────────────────────────────────
    cards = []
    for entry, frames in entries:
        seq_idx = entry["seq_idx"]
        source  = entry.get("source_image", "")
        p_stored = entry.get("p_animal", None)

        frame_data = []
        for f in frames:
            tensor  = _transform(Image.open(f).convert("RGB"))
            t_score = _score(teacher, tensor)
            s_score = _score(student, tensor)
            b64     = _b64(f)
            frame_data.append((f.name, t_score, s_score, b64))

        avg_teacher = sum(d[1] for d in frame_data) / len(frame_data)
        avg_student = sum(d[2] for d in frame_data) / len(frame_data)
        cards.append((avg_student, seq_idx, source, frame_data, avg_teacher, p_stored))

    # Sort by student p(animal) descending — worst offenders first
    cards.sort(reverse=True)

    # ── Build HTML ────────────────────────────────────────────────────────────
    def score_color(p):
        if p >= 0.7: return "#e03c3c"
        if p >= 0.4: return "#e09a3c"
        return "#4caf50"

    html_cards = []
    for avg_s, seq_idx, source, frame_data, avg_t, p_stored in cards:
        src_label = Path(source).name if source else f"seq_{seq_idx:05d}"
        n_frames  = len(frame_data)

        frame_imgs = ""
        for fname, t_sc, s_sc, b64 in frame_data:
            frame_imgs += f"""
            <div class="frame">
              <img src="data:image/jpeg;base64,{b64}" title="{fname}">
              <div class="scores">
                <span style="color:{score_color(s_sc)}">BNN {s_sc:.3f}</span>
                <span style="color:{score_color(t_sc)}">Teacher {t_sc:.3f}</span>
              </div>
            </div>"""

        html_cards.append(f"""
    <div class="card">
      <div class="header">
        <span class="seq">seq_{seq_idx:05d}</span>
        <span class="src">{src_label}</span>
        <span class="avg">
          BNN avg: <b style="color:{score_color(avg_s)}">{avg_s:.3f}</b> &nbsp;|&nbsp;
          Teacher avg: <b style="color:{score_color(avg_t)}">{avg_t:.3f}</b>
          &nbsp;({n_frames} frame{'s' if n_frames>1 else ''})
        </span>
      </div>
      <div class="frames">{frame_imgs}</div>
    </div>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hard Night Blank Gallery</title>
<style>
  body {{ background:#111; color:#ddd; font-family:monospace; padding:16px; }}
  h2   {{ font-size:15px; margin-bottom:4px; }}
  .meta {{ font-size:11px; color:#888; margin-bottom:20px; }}
  .card {{ background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px;
           padding:10px; margin-bottom:14px; }}
  .header {{ display:flex; gap:16px; align-items:baseline; margin-bottom:8px;
             font-size:12px; flex-wrap:wrap; }}
  .seq  {{ color:#aaa; min-width:90px; }}
  .src  {{ color:#666; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .avg  {{ font-size:12px; }}
  .frames {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .frame  {{ display:flex; flex-direction:column; align-items:center; }}
  .frame img {{ width:160px; height:160px; border-radius:3px; display:block; }}
  .scores {{ font-size:10px; margin-top:3px; display:flex; flex-direction:column;
             align-items:center; gap:1px; }}
</style>
</head>
<body>
<h2>Hard Night Blank Gallery</h2>
<p class="meta">
  {len(cards)} sequences &nbsp;|&nbsp;
  Student: {Path(args.student).name} ({student_acc:.1f}% val acc) &nbsp;|&nbsp;
  Teacher: {Path(args.teacher).name} &nbsp;|&nbsp;
  Sorted by BNN p(animal) descending — worst offenders first &nbsp;|&nbsp;
  <span style="color:#e03c3c">■</span> ≥0.7 &nbsp;
  <span style="color:#e09a3c">■</span> 0.4–0.7 &nbsp;
  <span style="color:#4caf50">■</span> &lt;0.4
</p>
{''.join(html_cards)}
</body>
</html>"""

    Path(args.out).write_text(html)
    print(f"\nDone → {args.out}")


if __name__ == "__main__":
    main()
