"""
gradcam.py
==========
Visualise which image regions drive the BNN's animal/blank decision
using Grad-CAM on the last convolutional feature map (bn4, 28×28 spatial).

The heatmap shows warm colours (red/yellow) where the model's "animal"
confidence is most sensitive — effectively, where it "sees" the animal.

Because Conv2-4 are binary (STE gradients), the signal is noisier than
a float32 network but still spatially meaningful.

Usage:
  # Single image, default checkpoint
  python project/software_training/gradcam.py photo.jpg

  # Batch, specific checkpoint, custom output dir
  python project/software_training/gradcam.py *.jpg \\
      --checkpoint project/bnn_distilled_876pct.pth \\
      --out-dir project/gradcam_outputs

  # Ensemble average (two checkpoints)
  python project/software_training/gradcam.py photo.jpg \\
      --checkpoint project/bnn_baseline_871pct.pth \\
      --ensemble   project/bnn_distilled_876pct.pth
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from bnn_serengeti2 import BNNClassifier, _transform, DEVICE, _NONBLANK_IDX, CHECKPOINT


# ── Grad-CAM core ─────────────────────────────────────────────────────────────
def _gradcam(model: torch.nn.Module, img_tensor: torch.Tensor,
             target_class: int, layer: str = "bn3") -> np.ndarray:
    """
    Returns a [H, W] numpy heatmap in [0, 1] showing which spatial regions
    most influenced the target class score.

    layer options (coarser → finer spatial resolution):
      bn4  28×28 — deepest semantic features
      bn3  56×56 — good balance of semantics + spatial precision  (default)
      bn2 112×112 — finest spatial detail, lower-level features

    Grad-CAM formula: ReLU( sum_c( mean_pool(dScore/dA_c) * A_c ) )
    """
    activations: dict = {}
    gradients:   dict = {}

    def _fwd(module, inp, out): activations["feat"] = out   # noqa: ARG001
    def _bwd(module, gi, go):  gradients["grad"]  = go[0]  # noqa: ARG001

    target = getattr(model, layer)
    fwd_hook = target.register_forward_hook(_fwd)
    bwd_hook = target.register_full_backward_hook(_bwd)

    try:
        model.zero_grad()
        score = model(img_tensor)[0, target_class]
        score.backward()
    finally:
        fwd_hook.remove()
        bwd_hook.remove()

    feat    = activations["feat"][0]          # [C, 28, 28]
    grad    = gradients["grad"][0]            # [C, 28, 28]
    weights = grad.mean(dim=(1, 2))           # [C] — spatial average of gradients
    cam     = (weights[:, None, None] * feat).sum(dim=0)  # [28, 28]
    cam     = F.relu(cam).detach().cpu().numpy()

    # Taper the two outermost rows/cols to suppress zero-padding artifacts.
    # Binary convs binarize padded zeros to +1, creating spurious edge activations.
    for k, w in enumerate([0.0, 0.35]):
        cam[k, :]      *= w;  cam[-(k+1), :] *= w
        cam[:, k]      *= w;  cam[:, -(k+1)] *= w

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam


def _overlay(original_pil: Image.Image, cam: np.ndarray,
             alpha: float = 0.55) -> Image.Image:
    """Blend a jet-coloured Grad-CAM heatmap over the original image."""
    h, w = original_pil.size[1], original_pil.size[0]
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32) / 255.0

    # Jet colormap: blue→cyan→green→yellow→red
    heatmap_rgb = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    orig_arr    = np.array(original_pil.convert("RGB"), dtype=np.float32)

    blended = alpha * heatmap_rgb + (1 - alpha) * orig_arr
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ── HTML slider comparison ────────────────────────────────────────────────────
def _to_b64(pil_img: Image.Image, size: int = 448) -> str:
    buf = io.BytesIO()
    pil_img.resize((size, size), Image.LANCZOS).save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _save_compare_html(original_pil: Image.Image, heatmap_pil: Image.Image,
                        out_path: str, title: str, p_animal: float,
                        decision: str) -> None:
    orig_b64 = _to_b64(original_pil)
    heat_b64 = _to_b64(heatmap_pil)
    colour   = "#e03c3c" if decision == "ANIMAL" else "#3cb87a"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Grad-CAM — {title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #111; color: #ddd; font-family: monospace;
  display: flex; flex-direction: column; align-items: center; padding: 20px;
}}
h2 {{ margin-bottom: 4px; font-size: 16px; }}
.meta {{ font-size: 13px; color: #aaa; margin-bottom: 14px; }}
.verdict {{ color: {colour}; font-weight: bold; }}
.wrap {{
  position: relative; width: 448px; height: 448px;
  cursor: col-resize; user-select: none; border: 1px solid #333;
}}
.wrap img {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: block;
}}
#overlay {{ clip-path: inset(0 50% 0 0); }}
.divider {{
  position: absolute; top: 0; left: 50%;
  width: 2px; height: 100%; background: white;
  transform: translateX(-50%); pointer-events: none;
}}
.handle {{
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 36px; height: 36px; border-radius: 50%;
  background: white; box-shadow: 0 2px 8px rgba(0,0,0,.6);
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; pointer-events: none; color: #333;
}}
.labels {{
  position: absolute; bottom: 8px; width: 100%;
  display: flex; justify-content: space-between; padding: 0 10px;
  pointer-events: none;
}}
.lbl {{
  background: rgba(0,0,0,.65); padding: 2px 8px;
  border-radius: 4px; font-size: 11px; color: #eee;
}}
p.hint {{ margin-top: 10px; font-size: 11px; color: #666; }}
</style>
</head>
<body>
<h2>Grad-CAM — {title}</h2>
<p class="meta">
  Decision: <span class="verdict">{decision}</span>
  &nbsp;|&nbsp; p(animal) = {p_animal:.3f}
</p>
<div class="wrap" id="wrap">
  <img src="data:image/jpeg;base64,{heat_b64}" alt="heatmap">
  <img src="data:image/jpeg;base64,{orig_b64}" alt="original" id="overlay">
  <div class="divider" id="div"></div>
  <div class="handle" id="handle">⇔</div>
  <div class="labels">
    <span class="lbl">Grad-CAM</span>
    <span class="lbl">Original</span>
  </div>
</div>
<p class="hint">Drag slider left/right to compare heatmap with original image</p>
<script>
const wrap = document.getElementById('wrap');
const overlay = document.getElementById('overlay');
const div = document.getElementById('div');
const handle = document.getElementById('handle');
let active = false;

function setPos(clientX) {{
  const r = wrap.getBoundingClientRect();
  const pct = Math.max(0, Math.min(100, (clientX - r.left) / r.width * 100));
  overlay.style.clipPath = `inset(0 ${{(100 - pct).toFixed(1)}}% 0 0)`;
  div.style.left = handle.style.left = pct + '%';
}}

wrap.addEventListener('mousedown',  e => {{ active = true; setPos(e.clientX); }});
window.addEventListener('mouseup',  () => active = false);
window.addEventListener('mousemove', e => {{ if (active) setPos(e.clientX); }});
wrap.addEventListener('touchstart', e => {{ active = true; setPos(e.touches[0].clientX); }}, {{passive:true}});
window.addEventListener('touchend', () => active = false);
window.addEventListener('touchmove', e => {{ if (active) setPos(e.touches[0].clientX); }}, {{passive:true}});
</script>
</body>
</html>"""

    Path(out_path).write_text(html, encoding="utf-8")


# ── Gallery ───────────────────────────────────────────────────────────────────
def rebuild_gallery(out_dir: Path) -> None:
    """
    Scan out_dir for all *_gradcam.html files and write gallery.html
    with every image embedded as an interactive slider card.
    Called automatically after each CLI run.
    """
    import re

    html_files = sorted(f for f in out_dir.glob("*_gradcam.html"))
    if not html_files:
        return

    cards_html = ""
    for hf in html_files:
        content = hf.read_text(encoding="utf-8")

        imgs = re.findall(r'src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)"', content)
        if len(imgs) < 2:
            continue
        heat_b64, orig_b64 = imgs[0], imgs[1]

        title_m   = re.search(r'<h2>Grad-CAM — (.+?)</h2>', content)
        verdict_m = re.search(r'class="verdict">(.+?)</span>', content)
        prob_m    = re.search(r'p\(animal\) = ([\d.]+)', content)

        title   = title_m.group(1)   if title_m   else hf.stem
        verdict = verdict_m.group(1) if verdict_m else "?"
        prob    = prob_m.group(1)    if prob_m    else "?"
        colour  = "#e03c3c" if verdict == "ANIMAL" else "#3cb87a"

        cards_html += f"""
  <div class="card">
    <div class="card-title">{title}</div>
    <div class="verdict" style="color:{colour}">{verdict} &nbsp; p={prob}</div>
    <div class="wrap">
      <img src="data:image/jpeg;base64,{heat_b64}">
      <img src="data:image/jpeg;base64,{orig_b64}" class="ov">
      <div class="dv"></div>
      <div class="hd">⇔</div>
      <div class="lbls">
        <span class="lbl">Heatmap</span>
        <span class="lbl">Original</span>
      </div>
    </div>
  </div>"""

    n = len(html_files)
    gallery = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Grad-CAM Gallery ({n} images)</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #111; color: #ddd; font-family: monospace; padding: 20px;
}}
h1 {{ font-size: 18px; margin-bottom: 4px; }}
.sub {{ font-size: 12px; color: #666; margin-bottom: 20px; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}}
.card {{
  background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px;
  padding: 10px; display: flex; flex-direction: column; gap: 5px;
}}
.card-title {{ font-size: 12px; color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.verdict {{ font-size: 13px; font-weight: bold; }}
.wrap {{
  position: relative; width: 100%; aspect-ratio: 1;
  cursor: col-resize; user-select: none; border-radius: 4px; overflow: hidden;
}}
.wrap img {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  display: block; object-fit: cover;
}}
.wrap .ov {{ clip-path: inset(0 50% 0 0); }}
.dv {{
  position: absolute; top: 0; left: 50%;
  width: 2px; height: 100%; background: white;
  transform: translateX(-50%); pointer-events: none;
}}
.hd {{
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 30px; height: 30px; border-radius: 50%;
  background: white; box-shadow: 0 1px 6px rgba(0,0,0,.6);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; pointer-events: none; color: #333;
}}
.lbls {{
  position: absolute; bottom: 6px; width: 100%;
  display: flex; justify-content: space-between; padding: 0 7px;
  pointer-events: none;
}}
.lbl {{
  background: rgba(0,0,0,.65); padding: 1px 6px;
  border-radius: 3px; font-size: 10px; color: #eee;
}}
</style>
</head>
<body>
<h1>Grad-CAM Gallery</h1>
<p class="sub">{n} image{'' if n == 1 else 's'} — drag any slider to compare heatmap with original</p>
<div class="grid">
{cards_html}
</div>
<script>
document.querySelectorAll('.wrap').forEach(wrap => {{
  const ov     = wrap.querySelector('.ov');
  const dv     = wrap.querySelector('.dv');
  const hd     = wrap.querySelector('.hd');
  let active   = false;

  function setPos(clientX) {{
    const r   = wrap.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, (clientX - r.left) / r.width * 100));
    ov.style.clipPath  = `inset(0 ${{(100 - pct).toFixed(1)}}% 0 0)`;
    dv.style.left = hd.style.left = pct + '%';
  }}

  wrap.addEventListener('mousedown',  e => {{ active = true;  setPos(e.clientX); }});
  wrap.addEventListener('touchstart', e => {{ active = true;  setPos(e.touches[0].clientX); }}, {{passive:true}});
  window.addEventListener('mouseup',  () => active = false);
  window.addEventListener('touchend', () => active = false);
  window.addEventListener('mousemove', e => {{ if (active) setPos(e.clientX); }});
  window.addEventListener('touchmove', e => {{ if (active) setPos(e.touches[0].clientX); }}, {{passive:true}});
}});
</script>
</body>
</html>"""

    gallery_path = out_dir / "gallery.html"
    gallery_path.write_text(gallery, encoding="utf-8")
    print(f"\n  Gallery → {gallery_path}  ({n} images)")


# ── Public entry point ────────────────────────────────────────────────────────
def run(image_path: str, checkpoints: list[str],
        out_path: str | None = None, threshold: float = 0.5,
        layer: str = "bn3") -> Image.Image:

    # Load model(s)
    models = []
    for ckpt_path in checkpoints:
        m = BNNClassifier()
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        m.to(DEVICE).eval()
        models.append(m)

    original = Image.open(image_path).convert("RGB")
    img_t    = _transform(original).unsqueeze(0).to(DEVICE)

    # Prediction (no grad needed here)
    with torch.no_grad():
        probs = sum(torch.softmax(m(img_t), dim=1) for m in models) / len(models)
    p_animal = probs[0, _NONBLANK_IDX].item()
    decision = "ANIMAL" if p_animal >= threshold else "BLANK"

    print(f"  {Path(image_path).name:<30s}  {decision}  p(animal)={p_animal:.3f}")

    # Grad-CAM: average heatmaps across ensemble members
    cams = []
    for m in models:
        cam = _gradcam(m, img_t.requires_grad_(False), _NONBLANK_IDX, layer)
        cams.append(cam)
    cam_avg = np.mean(cams, axis=0)

    # Overlay on original (resize to 224 for consistency)
    vis = _overlay(original.resize((224, 224), Image.LANCZOS), cam_avg)

    # Annotate corner with decision + probability
    label = f"{decision}  {p_animal:.2f}"
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(vis)
        colour = (220, 60, 60) if decision == "ANIMAL" else (60, 180, 60)
        draw.rectangle([0, 0, len(label) * 7 + 6, 16], fill=(0, 0, 0, 160))
        draw.text((3, 2), label, fill=colour)
    except Exception:
        pass

    if out_path is None:
        out_path = Path(image_path).stem + "_gradcam.jpg"
    vis.save(out_path, "JPEG", quality=92)

    html_path = str(Path(out_path).with_suffix(".html"))
    _save_compare_html(original.resize((224, 224), Image.LANCZOS), vis,
                       html_path, Path(image_path).name, p_animal, decision)
    print(f"  → {out_path}")
    print(f"  → {html_path}  (open in browser for slider comparison)")
    return vis


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM heatmap for BNN wildlife classifier")
    parser.add_argument("images", nargs="+", metavar="image.jpg",
                        help="One or more input images")
    parser.add_argument("--checkpoint", default=CHECKPOINT, metavar="CKPT")
    parser.add_argument("--ensemble",   default=None, metavar="CKPT",
                        help="Second checkpoint to average with primary")
    parser.add_argument("--out-dir",    default=None, metavar="DIR",
                        help="Output directory (default: same dir as input)")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--layer", default="bn3",
                        choices=["bn2", "bn3", "bn4"],
                        help="Target layer for Grad-CAM: bn2=112×112, bn3=56×56 (default), bn4=28×28")
    args = parser.parse_args()

    checkpoints = [args.checkpoint]
    if args.ensemble:
        checkpoints.append(args.ensemble)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint(s): {[Path(c).name for c in checkpoints]}")
    print(f"Threshold    : {args.threshold}\n")

    for img_path in args.images:
        p    = Path(img_path).resolve()
        stem = f"{p.parents[1].name}_{p.parent.name}_{p.stem}"
        out  = str((out_dir / f"{stem}_gradcam.jpg") if out_dir
                   else p.parent / f"{p.stem}_gradcam.jpg")
        run(img_path, checkpoints, out, args.threshold, args.layer)

    if out_dir:
        rebuild_gallery(out_dir)
