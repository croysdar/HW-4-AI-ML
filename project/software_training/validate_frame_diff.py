"""
validate_frame_diff.py
======================
Cheaply validates the frame-differencing concept WITHOUT retraining.

For each blank sequence in data_sequences/blank/:
  1. Estimates the background as the pixel-wise mean of all frames in the sequence
  2. Computes |frame_t - background| as the motion residual
  3. Blends: input = alpha * residual + (1 - alpha) * frame_t
  4. Runs the current (single-frame) model on the blended input
  5. Saves a 3-panel Grad-CAM HTML: original | frame-diff | blended

If the model scores lower p(animal) on diff/blended inputs for blank sequences,
the frame-difference approach is directionally sound even without retraining.

Usage:
  python project/software_training/validate_frame_diff.py \\
      --checkpoint project/bnn_serengeti2.pth \\
      --seq-dir project/data_sequences \\
      --alpha 0.7 \\
      --out-dir project/gradcam_framediff
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from bnn_serengeti2 import BNNClassifier, _transform, _NONBLANK_IDX, DEVICE, CHECKPOINT
from gradcam import _gradcam, _overlay, _save_compare_html, rebuild_gallery, _to_b64

import base64, io, json, re


def _load_model(ckpt_path: str) -> torch.nn.Module:
    m = BNNClassifier()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return m.to(DEVICE).eval()


def _background_mean(frame_paths: list[Path]) -> np.ndarray:
    """Pixel-wise mean of all frames as float32 [H, W, 3] in [0, 255]."""
    arrays = [np.array(Image.open(f).convert("RGB").resize((224, 224)), dtype=np.float32)
              for f in frame_paths]
    return np.mean(arrays, axis=0)


def _diff_image(frame: np.ndarray, background: np.ndarray,
                alpha: float) -> np.ndarray:
    """
    Compute blended input: alpha * |frame - background| + (1 - alpha) * frame
    All values in [0, 255] float32.
    """
    residual = np.abs(frame - background)
    blended  = alpha * residual + (1 - alpha) * frame
    return np.clip(blended, 0, 255)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """numpy [H,W,3] uint8-range → normalised tensor [1,3,H,W]."""
    pil = Image.fromarray(arr.astype(np.uint8))
    return _transform(pil).unsqueeze(0).to(DEVICE)


def _p_animal(model: torch.nn.Module, tensor: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0, _NONBLANK_IDX].item()


def _save_three_panel_html(orig_pil: Image.Image, diff_pil: Image.Image,
                            blend_pil: Image.Image, out_path: str,
                            p_orig: float, p_diff: float, p_blend: float,
                            title: str, alpha: float) -> None:
    """Three side-by-side Grad-CAM heatmaps for one sequence frame."""
    orig_b64  = _to_b64(orig_pil)
    diff_b64  = _to_b64(diff_pil)
    blend_b64 = _to_b64(blend_pil)

    def card(b64, label, p):
        colour = "#e03c3c" if p >= 0.5 else "#3cb87a"
        return f"""
  <div class="card">
    <div class="card-title">{label}</div>
    <div class="verdict" style="color:{colour}">p(animal)={p:.3f}</div>
    <img src="data:image/jpeg;base64,{b64}">
  </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Frame-Diff Validation — {title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background:#111; color:#ddd; font-family:monospace; padding:20px; }}
h2 {{ font-size:16px; margin-bottom:4px; }}
.meta {{ font-size:12px; color:#888; margin-bottom:16px; }}
.row {{ display:flex; gap:16px; flex-wrap:wrap; }}
.card {{
  background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px;
  padding:10px; display:flex; flex-direction:column; gap:6px; width:280px;
}}
.card-title {{ font-size:13px; font-weight:bold; }}
.verdict {{ font-size:12px; }}
.card img {{ width:100%; border-radius:4px; display:block; }}
</style>
</head>
<body>
<h2>Frame-Diff Validation — {title}</h2>
<p class="meta">alpha={alpha}  |  lower p(animal) on diff/blended = concept validated</p>
<div class="row">
  {card(orig_b64,  'Original (Grad-CAM)',       p_orig)}
  {card(diff_b64,  'Frame diff (Grad-CAM)',      p_diff)}
  {card(blend_b64, f'Blended α={alpha} (Grad-CAM)', p_blend)}
</div>
</body>
</html>"""
    Path(out_path).write_text(html, encoding="utf-8")


def run_sequence(seq_dir: Path, model: torch.nn.Module,
                 alpha: float, out_dir: Path, seq_label: str,
                 seq_idx: int, layer: str = "bn3") -> dict:
    """Process one sequence. Returns summary dict."""
    frames = sorted(seq_dir.glob("frame_*.jpg"))
    if not frames:
        return {}

    background = _background_mean(frames)
    results = []

    for i, f in enumerate(frames):
        frame_arr = np.array(Image.open(f).convert("RGB").resize((224, 224)),
                             dtype=np.float32)
        diff_arr  = _diff_image(frame_arr, background, alpha=1.0)
        blend_arr = _diff_image(frame_arr, background, alpha=alpha)

        t_orig  = _to_tensor(frame_arr)
        t_diff  = _to_tensor(diff_arr)
        t_blend = _to_tensor(blend_arr)

        p_orig  = _p_animal(model, t_orig)
        p_diff  = _p_animal(model, t_diff)
        p_blend = _p_animal(model, t_blend)

        # Grad-CAM heatmaps on each input
        cam_orig  = _gradcam(model, t_orig,  _NONBLANK_IDX, layer)
        cam_diff  = _gradcam(model, t_diff,  _NONBLANK_IDX, layer)
        cam_blend = _gradcam(model, t_blend, _NONBLANK_IDX, layer)

        orig_pil  = Image.fromarray(frame_arr.astype(np.uint8))
        diff_pil  = Image.fromarray(diff_arr.astype(np.uint8))
        blend_pil = Image.fromarray(blend_arr.astype(np.uint8))

        vis_orig  = _overlay(orig_pil,  cam_orig)
        vis_diff  = _overlay(diff_pil,  cam_diff)
        vis_blend = _overlay(blend_pil, cam_blend)

        stem     = f"{seq_label}_seq{seq_idx:03d}_frame{i+1:02d}"
        out_html = str(out_dir / f"{stem}.html")
        _save_three_panel_html(vis_orig, vis_diff, vis_blend, out_html,
                               p_orig, p_diff, p_blend,
                               f"{seq_label} seq{seq_idx:03d} frame{i+1}",
                               alpha)
        results.append((p_orig, p_diff, p_blend))

        delta = p_diff - p_orig
        arrow = "↓" if delta < -0.05 else ("↑" if delta > 0.05 else "→")
        print(f"    frame {i+1}  orig={p_orig:.3f}  diff={p_diff:.3f} {arrow}  "
              f"blend={p_blend:.3f}  [{f.name}]")

    return {"frames": results}


def main():
    parser = argparse.ArgumentParser(
        description="Validate frame-differencing concept on existing sequences")
    parser.add_argument("--checkpoint", default=CHECKPOINT, metavar="CKPT")
    parser.add_argument("--seq-dir",   default="project/data_sequences", metavar="DIR")
    parser.add_argument("--alpha",     type=float, default=0.7,
                        help="Blend weight for diff (1.0=pure diff, 0.0=original, default 0.7)")
    parser.add_argument("--n-seqs",    type=int, default=10,
                        help="Number of sequences per class to test (default: 10)")
    parser.add_argument("--layer",     default="bn3", choices=["bn2", "bn3", "bn4"])
    parser.add_argument("--out-dir",   default="project/gradcam_framediff", metavar="DIR")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = _load_model(args.checkpoint)

    index = json.loads((Path(args.seq_dir) / "seq_index.json").read_text())

    blank_entries  = [e for e in index if e["label"] == "blank"][:args.n_seqs]
    animal_entries = [e for e in index if e["label"] == "animal"][:args.n_seqs]

    print(f"\nalpha={args.alpha}  (diff weight)  |  ↓ = diff lowered p(animal)\n")

    all_orig_blank, all_diff_blank = [], []
    all_orig_animal, all_diff_animal = [], []

    for label, entries, orig_list, diff_list in [
        ("blank",  blank_entries,  all_orig_blank,  all_diff_blank),
        ("animal", animal_entries, all_orig_animal, all_diff_animal),
    ]:
        print(f"{'═'*60}")
        print(f"  {label.upper()} sequences  (n={len(entries)})")
        print(f"{'═'*60}")
        for entry in entries:
            seq_path = Path(args.seq_dir) / label / f"seq_{entry['seq_idx']:05d}"
            print(f"\n  seq_{entry['seq_idx']:05d}  loc={entry.get('location','?')}")
            result = run_sequence(seq_path, model, args.alpha, out_dir,
                                  label, entry["seq_idx"], args.layer)
            for p_o, p_d, p_b in result.get("frames", []):
                orig_list.append(p_o)
                diff_list.append(p_d)

    # ── Summary ───────────────────────────────────────────────────────────────
    def stats(vals):
        a = np.array(vals)
        return f"mean={a.mean():.3f}  FAR@0.5={100*(a>=0.5).mean():.1f}%"

    print(f"\n{'═'*60}")
    print(f"  SUMMARY (alpha={args.alpha})")
    print(f"{'═'*60}")
    print(f"  BLANK   original : {stats(all_orig_blank)}")
    print(f"  BLANK   diff     : {stats(all_diff_blank)}")
    print(f"  ANIMAL  original : {stats(all_orig_animal)}")
    print(f"  ANIMAL  diff     : {stats(all_diff_animal)}")
    print(f"{'═'*60}")
    print(f"\n  HTML frames → {out_dir}/")
    print(f"  Open any .html to see 3-panel comparison (original / diff / blended)")


if __name__ == "__main__":
    main()
