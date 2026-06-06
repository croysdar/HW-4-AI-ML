#!/usr/bin/env python3
"""
run_hw_inference_8macro.py
==========================
Hardware inference co-simulation for the 8-SRAM 256-bit bnn_top variant.

Identical logic to run_hw_inference.py (1-macro) with three differences:

  1. RTL sources point to sram_8macro_experiment files.
  2. SRAM_LOGICAL_DEPTH = 256  (8 banks × 32-bit wide × 256 rows = 256 logical words)
  3. filters_per_batch updated:
       conv2: floor(256/2) = 128 → capped at 64 out_ch  → 64 (all in one batch)
       conv3: floor(256/3) = 85  → capped at 128 out_ch → 85 (2 batches)
       conv4: floor(256/5) = 51  → capped at 256 out_ch → 51 (6 batches)

Usage
-----
  python run_hw_inference_8macro.py --image <path>
  python run_hw_inference_8macro.py --n-images 20
  python run_hw_inference_8macro.py --n-images 20 --checkpoint project/bnn_serengeti2.pth
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR   = Path(__file__).resolve().parent           # sram_8macro_experiment/
M4_DIR     = THIS_DIR.parent                           # m4/
PROJECT    = M4_DIR.parent                             # project/
REPO_ROOT  = PROJECT.parent                            # HW-4-AI-ML/

DEFAULT_CHECKPOINT = PROJECT / "bnn_serengeti2.pth"
TEST_DIR           = PROJECT / "data_20k" / "test"

RTL_SRCS = [
    THIS_DIR / "sim" / "tb_hw_inference_8macro.sv",
    THIS_DIR / "top_sram8macro.sv",
    M4_DIR   / "rtl" / "compute_core.sv",
    M4_DIR   / "rtl" / "interface.sv",
    THIS_DIR / "sky130_sram_1kbyte_1rw1r_32x256_8_behav.sv",
]

SIM_BINARY = THIS_DIR / "hw_inference_8macro.vvp"

# ---------------------------------------------------------------------------
# Import BNN model
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT))
from bnn_serengeti2 import BNNClassifier, _transform  # noqa: E402


def load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _transform(img).unsqueeze(0)


def load_model(checkpoint: str) -> BNNClassifier:
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = BNNClassifier()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    return model


def extract_layer_params(model: BNNClassifier, layer_name: str) -> dict:
    conv = getattr(model, layer_name)
    bn   = getattr(model, layer_name.replace("conv", "bn"))
    with torch.no_grad():
        w_real = conv.weight.data
        w_bin  = torch.sign(w_real)
        w_bin  = torch.where(w_bin == 0, torch.ones_like(w_bin), w_bin)
    return {
        "weights_bin": w_bin.numpy().astype(np.int8),
        "alpha":       conv.alpha.data.numpy().astype(np.float32),
        "bn_mean":     bn.running_mean.numpy().astype(np.float32),
        "bn_var":      bn.running_var.numpy().astype(np.float32),
        "bn_weight":   bn.weight.data.numpy().astype(np.float32),
        "bn_bias":     bn.bias.data.numpy().astype(np.float32),
    }


def binarize_np(x: np.ndarray) -> np.ndarray:
    b = np.sign(x).astype(np.int8)
    b[b == 0] = 1
    return b


def to_bits(binary_vec: np.ndarray) -> np.ndarray:
    bits = ((binary_vec + 1) >> 1).astype(np.uint8)
    pad  = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little")


def pack_to_256bit_words(binary_vec: np.ndarray) -> list[bytes]:
    byte_arr = to_bits(binary_vec)
    pad = (-len(byte_arr)) % 32
    if pad:
        byte_arr = np.concatenate([byte_arr, np.zeros(pad, dtype=np.uint8)])
    return [bytes(byte_arr[i:i+32]) for i in range(0, len(byte_arr), 32)]


def word_to_hex(word_bytes: bytes) -> str:
    return word_bytes[::-1].hex()


def compute_thresholds(params: dict) -> np.ndarray:
    eps   = 1e-5
    mean  = params["bn_mean"]
    var   = params["bn_var"]
    gamma = params["bn_weight"]
    beta  = params["bn_bias"]
    alpha = params["alpha"]
    with np.errstate(divide="ignore", invalid="ignore"):
        thresh = np.where(
            np.abs(alpha) < 1e-9,
            np.full_like(alpha, np.inf),
            (mean - beta * np.sqrt(var + eps) / (gamma + 1e-12)) / alpha
        )
    return thresh.astype(np.float32)


def pytorch_full_forward(model: BNNClassifier,
                         img_tensor: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    pred       = int(probs.argmax().item())
    confidence = float(probs[pred].item()) * 100.0
    return pred, confidence


def extract_receptive_field(act_map: np.ndarray,
                             out_row: int, out_col: int,
                             stride: int, pad: int,
                             kH: int, kW: int) -> np.ndarray:
    in_ch, H, W = act_map.shape
    field = np.full((in_ch, kH, kW), -1, dtype=np.int8)
    for ki in range(kH):
        for kj in range(kW):
            r = out_row * stride - pad + ki
            c = out_col * stride - pad + kj
            if 0 <= r < H and 0 <= c < W:
                field[:, ki, kj] = act_map[:, r, c]
    return field.reshape(-1)


# ---------------------------------------------------------------------------
# Weight / stimulus file writers
# ---------------------------------------------------------------------------
# 8-macro: 256 logical addresses (vs 32 for 1-macro).
SRAM_LOGICAL_DEPTH = 256

def write_weight_batch(wt_file, filter_weights: list, n_tiles: int, n_beats: int) -> None:
    n_filters = len(filter_weights)
    n_words   = n_filters * n_beats
    wt_file.write(f"LOAD {SRAM_LOGICAL_DEPTH}\n")
    for f_words in filter_weights:
        for word_bytes in f_words:
            wt_file.write(word_to_hex(word_bytes) + "\n")
    zero_word = "00" * 32
    for _ in range(SRAM_LOGICAL_DEPTH - n_words):
        wt_file.write(zero_word + "\n")


def write_stimulus_batch(stim_file, tiles: list) -> None:
    stim_file.write(f"# BATCH {len(tiles)}\n")
    for (cfg_beats, w_base, beats) in tiles:
        hex_parts = [word_to_hex(b) for b in beats]
        stim_file.write(f"{cfg_beats} {w_base} " + " ".join(hex_parts) + "\n")


def run_simulation(run_dir: Path) -> list[int]:
    result = subprocess.run(
        ["vvp", str(SIM_BINARY)],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
    )
    stdout = result.stdout + result.stderr
    if "HW_INFERENCE_DONE" not in stdout:
        print(f"    [SIM STDOUT]\n{stdout}")
        raise RuntimeError("vvp simulation did not complete successfully")
    results_path = run_dir / "hw_inference_results.txt"
    dots = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                dots.append(int(line))
    return dots


def compile_sim() -> None:
    print("[COMPILE] Checking RTL sources...")
    missing = [str(s) for s in RTL_SRCS if not s.exists()]
    if missing:
        raise FileNotFoundError("Missing RTL source(s):\n  " + "\n  ".join(missing))
    cmd = ["iverilog", "-g2012", "-o", str(SIM_BINARY)] + [str(s) for s in RTL_SRCS]
    print(f"[COMPILE] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("iverilog compilation failed")
    print(f"[COMPILE] OK → {SIM_BINARY}")


# ---------------------------------------------------------------------------
# Hardware BNN layer forward
# ---------------------------------------------------------------------------
def hw_bnn_layer_forward(
    layer_name:        str,
    params:            dict,
    act_map_bin:       np.ndarray,
    n_beats:           int,
    stride:            int,
    pad:               int,
    filters_per_batch: int,
    run_dir:           Path,
) -> np.ndarray:
    weights_bin = params["weights_bin"]
    out_ch, in_ch, kH, kW = weights_bin.shape
    in_H, in_W = act_map_bin.shape[1], act_map_bin.shape[2]
    out_H = (in_H + 2 * pad - kH) // stride + 1
    out_W = (in_W + 2 * pad - kW) // stride + 1
    n_spatial = out_H * out_W

    all_filter_words = []
    for f in range(out_ch):
        fvec  = weights_bin[f].reshape(-1)
        words = pack_to_256bit_words(fvec)
        assert len(words) == n_beats, (
            f"{layer_name} filter {f}: expected {n_beats} beats, got {len(words)}"
        )
        all_filter_words.append(words)

    print(f"    [{layer_name}] extracting {n_spatial} receptive fields ...")
    rf_beats = []
    for row in range(out_H):
        for col in range(out_W):
            rf_vec   = extract_receptive_field(act_map_bin, row, col, stride, pad, kH, kW)
            rf_words = pack_to_256bit_words(rf_vec)
            assert len(rf_words) == n_beats
            rf_beats.append(rf_words)

    n_batches = math.ceil(out_ch / filters_per_batch)
    all_dots  = []

    for batch_idx in range(n_batches):
        f_start = batch_idx * filters_per_batch
        f_end   = min(f_start + filters_per_batch, out_ch)
        batch_filters = list(range(f_start, f_end))
        n_f_this  = len(batch_filters)
        n_tiles_this = n_f_this * n_spatial

        print(f"    [{layer_name}] batch {batch_idx+1}/{n_batches}: "
              f"filters {f_start}–{f_end-1} ({n_tiles_this} tiles) ...", end="", flush=True)

        wt_path   = run_dir / "hw_inference_weights.txt"
        stim_path = run_dir / "hw_inference_stimulus.txt"

        with open(wt_path, "w") as wt_file:
            write_weight_batch(wt_file, [all_filter_words[f] for f in batch_filters],
                               n_tiles_this, n_beats)

        tiles = []
        for sp_idx in range(n_spatial):
            for fj, f_abs in enumerate(batch_filters):
                w_base = fj * n_beats
                tiles.append((n_beats, w_base, rf_beats[sp_idx]))

        with open(stim_path, "w") as stim_file:
            write_stimulus_batch(stim_file, tiles)

        batch_dots = run_simulation(run_dir)
        if len(batch_dots) != n_tiles_this:
            raise RuntimeError(
                f"{layer_name} batch {batch_idx}: expected {n_tiles_this} "
                f"results, got {len(batch_dots)}"
            )
        all_dots.extend(batch_dots)
        print(f" done")

    dot_arr      = np.array(all_dots, dtype=np.int32).reshape(n_spatial, out_ch)
    dot_map_flat = dot_arr.T
    dot_map      = dot_map_flat.reshape(out_ch, out_H, out_W)

    alpha  = params["alpha"][:, None, None]
    mean   = params["bn_mean"][:, None, None]
    var    = params["bn_var"][:, None, None]
    gamma  = params["bn_weight"][:, None, None]
    beta   = params["bn_bias"][:, None, None]
    eps    = 1e-5
    scaled = alpha * dot_map.astype(np.float32)
    y      = (scaled - mean) / np.sqrt(var + eps) * gamma + beta
    return np.where(y > 0, np.int8(1), np.int8(-1))


# ---------------------------------------------------------------------------
# Full hardware inference for one image
# ---------------------------------------------------------------------------
def hw_inference_one_image(img_path: str, model: BNNClassifier,
                            layer_params: dict, run_dir: Path) -> tuple[int, float]:
    img_tensor = load_image_tensor(img_path)

    with torch.no_grad():
        x = model.bn1(model.conv1(img_tensor))
    act2_bin = binarize_np(x[0].numpy())

    # 8-macro filters_per_batch: floor(SRAM_LOGICAL_DEPTH / n_beats), capped at out_ch
    print(f"  [conv2] 64 output channels, 2 beats/filter, "
          f"{min(SRAM_LOGICAL_DEPTH//2, 64)} filters/batch ...")
    act3_bin = hw_bnn_layer_forward(
        layer_name="conv2", params=layer_params["conv2"], act_map_bin=act2_bin,
        n_beats=2, stride=2, pad=1, filters_per_batch=min(SRAM_LOGICAL_DEPTH//2, 64),
        run_dir=run_dir,
    )

    print(f"  [conv3] 128 output channels, 3 beats/filter, "
          f"{min(SRAM_LOGICAL_DEPTH//3, 128)} filters/batch ...")
    act4_bin = hw_bnn_layer_forward(
        layer_name="conv3", params=layer_params["conv3"], act_map_bin=act3_bin,
        n_beats=3, stride=2, pad=1, filters_per_batch=min(SRAM_LOGICAL_DEPTH//3, 128),
        run_dir=run_dir,
    )

    print(f"  [conv4] 256 output channels, 5 beats/filter, "
          f"{min(SRAM_LOGICAL_DEPTH//5, 256)} filters/batch ...")
    act5_bin = hw_bnn_layer_forward(
        layer_name="conv4", params=layer_params["conv4"], act_map_bin=act4_bin,
        n_beats=5, stride=2, pad=1, filters_per_batch=min(SRAM_LOGICAL_DEPTH//5, 256),
        run_dir=run_dir,
    )

    feat = torch.from_numpy(act5_bin.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pooled = model.pool(feat)
        flat   = torch.flatten(pooled, 1)
        logits = model.fc(flat)
        probs  = torch.softmax(logits, dim=1)[0]
    pred       = int(probs.argmax().item())
    confidence = float(probs[pred].item()) * 100.0
    return pred, confidence


# ---------------------------------------------------------------------------
# Image helpers (identical to 1-macro version)
# ---------------------------------------------------------------------------
def is_night_image(path: str) -> bool:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    step_x = max(1, w // 15)
    step_y = max(1, h // 15)
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            r, g, b = img.getpixel((x, y))
            if r != g or g != b:
                return False
    return True


def collect_test_images(n_images=None, single_image=None, tod=None):
    if single_image:
        return [single_image]
    blank_dir    = TEST_DIR / "blank"
    nonblank_dir = TEST_DIR / "non_blank"
    if not blank_dir.exists() or not nonblank_dir.exists():
        raise FileNotFoundError(f"Test directories not found:\n  {blank_dir}\n  {nonblank_dir}")
    blank_imgs    = sorted(blank_dir.glob("*.jpg")) + sorted(blank_dir.glob("*.png"))
    nonblank_imgs = sorted(nonblank_dir.glob("*.jpg")) + sorted(nonblank_dir.glob("*.png"))
    if tod is not None:
        want_night = (tod == "night")
        print(f"[IMAGES] Filtering for {tod} images ...")
        blank_imgs    = [p for p in blank_imgs    if is_night_image(str(p)) == want_night]
        nonblank_imgs = [p for p in nonblank_imgs if is_night_image(str(p)) == want_night]
        print(f"[IMAGES] After filter: {len(blank_imgs)} blank, {len(nonblank_imgs)} non_blank")
    if n_images is None:
        return [str(p) for p in blank_imgs + nonblank_imgs]
    half      = n_images // 2
    extra     = n_images - half
    blanks    = random.sample(blank_imgs,    min(half,  len(blank_imgs)))
    nonblanks = random.sample(nonblank_imgs, min(extra, len(nonblank_imgs)))
    imgs      = [str(p) for p in blanks + nonblanks]
    random.shuffle(imgs)
    return imgs


def path_to_label(path: str) -> int:
    p = Path(path)
    if "non_blank" in p.parts or "non_blank" in p.parent.name:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BNN hardware inference co-simulation (8-macro)")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--image",    metavar="PATH")
    grp.add_argument("--n-images", metavar="N", type=int)
    parser.add_argument("--checkpoint", metavar="PATH", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tod", choices=["day", "night"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    if not shutil.which("iverilog"):
        print("ERROR: iverilog not found.  brew install icarus-verilog")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    compile_sim()

    print(f"\n[MODEL] Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint)
    layer_params = {
        "conv2": extract_layer_params(model, "conv2"),
        "conv3": extract_layer_params(model, "conv3"),
        "conv4": extract_layer_params(model, "conv4"),
    }
    for lname, p in layer_params.items():
        print(f"  {lname}: {p['weights_bin'].shape}")

    images = collect_test_images(n_images=args.n_images, single_image=args.image, tod=args.tod)
    print(f"\n[IMAGES] {len(images)} image(s) to process.")

    run_dir = THIS_DIR / "hw_inference_run_8macro"
    run_dir.mkdir(exist_ok=True)

    CLASS_NAMES = ["blank", "non_blank"]
    n_match = 0
    n_total = 0

    print()
    print("=" * 80)
    print(f"{'Image':<36}  {'ToD':>5}  {'GT':>8}  {'PyTorch':>8}  {'HW':>8}  {'Match':>6}")
    print("-" * 80)

    for img_path in images:
        n_total += 1
        img_name  = Path(img_path).name
        gt_label  = path_to_label(img_path)
        tod_label = "night" if is_night_image(img_path) else "day"
        gt_name   = CLASS_NAMES[gt_label]

        print(f"\n[IMAGE {n_total}/{len(images)}] {img_name}")
        img_tensor = load_image_tensor(img_path)
        pt_pred, pt_conf = pytorch_full_forward(model, img_tensor)
        pt_name = CLASS_NAMES[pt_pred]
        print(f"  PyTorch: {pt_name} ({pt_conf:.1f}%)")

        try:
            hw_pred, hw_conf = hw_inference_one_image(
                img_path=img_path, model=model,
                layer_params=layer_params, run_dir=run_dir,
            )
        except Exception as exc:
            print(f"  HW ERROR: {exc}")
            continue

        hw_name   = CLASS_NAMES[hw_pred]
        match     = (hw_pred == pt_pred)
        if match:
            n_match += 1
        match_str = "Y" if match else "N"
        print(f"  HW:    {hw_name} ({hw_conf:.1f}%)  Match: {match_str}")
        print(f"  {img_name:<36}  {tod_label:>5}  {gt_name:>8}  {pt_name:>8}  {hw_name:>8}  {match_str:>6}")

    print()
    print("=" * 72)
    print(f"SUMMARY")
    print(f"  Images processed  : {n_total}")
    print(f"  HW matches PyTorch: {n_match}/{n_total}  "
          f"({100.0*n_match/n_total if n_total else 0:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
