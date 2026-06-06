#!/usr/bin/env python3
"""
run_hw_inference.py
===================
Hardware inference co-simulation for the BNN accelerator chiplet.

Runs real wildlife camera images through the actual RTL simulation (iverilog +
vvp) and compares the result to a PyTorch floating-point reference inference,
validating that the hardware dot products produce the same binary decisions.

Usage
-----
  python run_hw_inference.py --image <path>
  python run_hw_inference.py --n-images 10
  python run_hw_inference.py --n-images 20 --checkpoint project/bnn_serengeti2.pth

Arguments
---------
  --image <path>         Path to a single image file to run.
  --n-images N           Pick N random images (half blank, half non_blank) from
                         the test set and run all of them.
  --checkpoint <path>    Path to the .pth checkpoint (default: project/bnn_serengeti2.pth
                         relative to two directories above this script).

Design notes
------------
  - iverilog is compiled once at startup; vvp is re-invoked per layer per batch.
  - The testbench reads hw_inference_weights.txt and hw_inference_stimulus.txt
    from the current working directory (the sim run directory).
  - Stimulus format:  "<cfg_beats> <w_base> <beat0_hex> [<beat1_hex> ...]"
    preceded by a batch header comment "# BATCH <n_tiles>".
  - Weight file format: "LOAD <n_words>" followed by n_words hex lines.
  - Results are read from hw_inference_results.txt.

Bit-packing convention
----------------------
  +1 activation → bit 1, -1 activation → bit 0.
  A 256-bit word is stored LSB-first in memory (chunk 0 = bits[31:0]).
  np.packbits with bitorder='little' maps the first element of a binary array
  to the LSB of the first byte, matching the hardware's chunk extraction:
    act_chunk = act_buf[chunk_ctr*32 +: 32]
  which reads bits 31:0 in chunk 0.

  Weight packing: weight tensor shape [out_ch, in_ch, kH, kW].  For each
  output channel we flatten [in_ch, kH, kW] into a 1-D binary vector of
  length in_ch * kH * kW bits, zero-pad to a multiple of 256, then split
  into 256-bit (32-byte) logical words.

  Beats per filter:
    conv2: 32*9 = 288 bits → 2 beats  (second beat zero-padded to 256)
    conv3: 64*9 = 576 bits → 3 beats  (third beat zero-padded to 256)
    conv4: 128*9=1152 bits → 5 beats  (fifth beat zero-padded to 256)
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
THIS_DIR   = Path(__file__).resolve().parent           # sram_1macro_experiment/
M4_DIR     = THIS_DIR.parent                           # m4/
PROJECT    = M4_DIR.parent                             # project/
REPO_ROOT  = PROJECT.parent                            # HW-4-AI-ML/

DEFAULT_CHECKPOINT = PROJECT / "bnn_serengeti2.pth"
TEST_DIR           = PROJECT / "data_20k" / "test"

# RTL source files used for iverilog compile
RTL_SRCS = [
    THIS_DIR / "tb_hw_inference.sv",
    THIS_DIR / "top_sram1macro.sv",
    THIS_DIR / "compute_core_narrow.sv",
    M4_DIR   / "rtl" / "interface.sv",
    M4_DIR   / "sram_256_experiment" / "sky130_sram_1kbyte_1rw1r_32x256_8_behav.sv",
]

# Output binary
SIM_BINARY = THIS_DIR / "hw_inference.vvp"

# ---------------------------------------------------------------------------
# Import BNN model from project/bnn_serengeti2.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT))
from bnn_serengeti2 import BNNClassifier, _transform   # noqa: E402


# ---------------------------------------------------------------------------
# Image transform (same as bnn_serengeti2 inference)
# ---------------------------------------------------------------------------
def load_image_tensor(path: str) -> torch.Tensor:
    """Load and preprocess one image → [1, 3, 224, 224] float tensor."""
    img = Image.open(path).convert("RGB")
    return _transform(img).unsqueeze(0)  # [1, 3, 224, 224]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint: str) -> BNNClassifier:
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = BNNClassifier()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Extract layer parameters
# ---------------------------------------------------------------------------
def extract_layer_params(model: BNNClassifier, layer_name: str) -> dict:
    """
    Return a dict with:
      weights_bin : np.ndarray int8 {+1,-1} shape [out_ch, in_ch, kH, kW]
      alpha       : np.ndarray float32 [out_ch]
      bn_mean     : np.ndarray float32 [out_ch]
      bn_var      : np.ndarray float32 [out_ch]
      bn_weight   : np.ndarray float32 [out_ch]
      bn_bias     : np.ndarray float32 [out_ch]
    """
    conv = getattr(model, layer_name)    # BinarizeConv2d
    bn   = getattr(model, layer_name.replace("conv", "bn"))  # BatchNorm2d

    with torch.no_grad():
        w_real = conv.weight.data          # [out, in, kH, kW]
        w_bin  = torch.sign(w_real)
        w_bin  = torch.where(w_bin == 0, torch.ones_like(w_bin), w_bin)

    return {
        "weights_bin": w_bin.numpy().astype(np.int8),   # {+1, -1}
        "alpha":       conv.alpha.data.numpy().astype(np.float32),
        "bn_mean":     bn.running_mean.numpy().astype(np.float32),
        "bn_var":      bn.running_var.numpy().astype(np.float32),
        "bn_weight":   bn.weight.data.numpy().astype(np.float32),
        "bn_bias":     bn.bias.data.numpy().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Binarization helpers
# ---------------------------------------------------------------------------
def binarize_np(x: np.ndarray) -> np.ndarray:
    """sign(x) with 0 → +1.  Returns int8 {+1, -1}."""
    b = np.sign(x).astype(np.int8)
    b[b == 0] = 1
    return b


def to_bits(binary_vec: np.ndarray) -> np.ndarray:
    """
    Convert a {+1,-1} vector of length N to a uint8 array of ceil(N/8) bytes.
    +1 → bit 1, -1 → bit 0.  Bit order: first element → LSB of first byte
    (bitorder='little'), matching hardware's chunk_ctr=0 → bits[31:0].
    Zero-pads to a multiple of 8 bits.
    """
    bits = ((binary_vec + 1) >> 1).astype(np.uint8)   # {+1,-1} → {1,0}
    pad  = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little")


def pack_to_256bit_words(binary_vec: np.ndarray) -> list[bytes]:
    """
    Pack a {+1,-1} vector into a list of 32-byte (256-bit) words.
    Zero-pads to a multiple of 256 bits.
    The first 32 bytes correspond to the first AXI beat (bits[255:0]),
    with the LSB of beat byte 0 = element 0 of binary_vec.
    """
    byte_arr = to_bits(binary_vec)  # ceil(N/8) bytes, LSB-first per byte
    # Pad to multiple of 32 bytes (256 bits)
    pad = (-len(byte_arr)) % 32
    if pad:
        byte_arr = np.concatenate([byte_arr, np.zeros(pad, dtype=np.uint8)])
    # Split into 32-byte chunks (each chunk = one 256-bit beat)
    words = []
    for i in range(0, len(byte_arr), 32):
        words.append(bytes(byte_arr[i:i+32]))
    return words


def word_to_hex(word_bytes: bytes) -> str:
    """
    Convert 32 bytes (256-bit word) to a 64-char hex string.
    The hardware stores chunk 0 = bits[31:0] = bytes[0..3].
    We emit the word as a 256-bit hex number: most-significant nibble first
    (i.e. bytes[31] ... bytes[0]), which is what $sscanf("%h", ...) reads
    into a SystemVerilog 256-bit variable (MSB = leftmost hex digit).
    """
    # bytes are LSB-first; reverse to get MSB-first for hex representation
    return word_bytes[::-1].hex()


# ---------------------------------------------------------------------------
# Threshold computation
#   threshold_c = (bn_mean[c] - bn_bias[c]*sqrt(bn_var[c]+1e-5)/bn_weight[c]) / alpha[c]
# ---------------------------------------------------------------------------
def compute_thresholds(params: dict) -> np.ndarray:
    """
    For each output channel c, the hardware dot product dot[c] passes binarize
    iff the final BatchNorm output y[c] > 0, which simplifies to:
        dot[c] > thresh[c]
    where thresh[c] = (bn_mean[c] - bn_bias[c]*sqrt(bn_var[c]+eps)/bn_weight[c]) / alpha[c]

    Returns float32 array of shape [out_ch].
    NOTE: This is only used for the *pure-Python* reference path; the hardware
    path drives the RTL directly and reads back integer dot products.
    """
    eps      = 1e-5
    mean     = params["bn_mean"]
    var      = params["bn_var"]
    gamma    = params["bn_weight"]
    beta     = params["bn_bias"]
    alpha    = params["alpha"]
    # Handle zero/near-zero alpha or gamma gracefully
    with np.errstate(divide="ignore", invalid="ignore"):
        thresh = np.where(
            np.abs(alpha) < 1e-9,
            np.full_like(alpha, np.inf),
            (mean - beta * np.sqrt(var + eps) / (gamma + 1e-12)) / alpha
        )
    return thresh.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch reference forward for a single image
# ---------------------------------------------------------------------------
def pytorch_full_forward(model: BNNClassifier,
                         img_tensor: torch.Tensor) -> tuple[int, float]:
    """
    Full PyTorch forward pass.  Returns (predicted_class, confidence_pct).
    class 0 = blank, class 1 = non_blank.
    """
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    pred       = int(probs.argmax().item())
    confidence = float(probs[pred].item()) * 100.0
    return pred, confidence


# ---------------------------------------------------------------------------
# Extract activation receptive field at one spatial position
# ---------------------------------------------------------------------------
def extract_receptive_field(act_map: np.ndarray,
                             out_row: int, out_col: int,
                             stride: int, pad: int,
                             kH: int, kW: int) -> np.ndarray:
    """
    Extract the receptive field of a kH×kW convolution centred at output
    position (out_row, out_col) from act_map [in_ch, H, W].
    Returns a flat int8 {+1,-1} array of length in_ch * kH * kW.
    Out-of-bounds pixels are treated as -1 (zero-padding → bit 0).
    """
    in_ch, H, W = act_map.shape
    field = np.full((in_ch, kH, kW), -1, dtype=np.int8)  # default = padded (-1)
    for ki in range(kH):
        for kj in range(kW):
            r = out_row * stride - pad + ki
            c = out_col * stride - pad + kj
            if 0 <= r < H and 0 <= c < W:
                field[:, ki, kj] = act_map[:, r, c]
    # Flatten in C-order: [in_ch, kH, kW] → [in_ch * kH * kW]
    return field.reshape(-1)


# ---------------------------------------------------------------------------
# Write weight file for one SRAM load batch
# ---------------------------------------------------------------------------
def write_weight_batch(wt_file,
                       filter_weights: list[list[bytes]],
                       n_tiles: int,
                       n_beats: int) -> None:
    """
    Write a LOAD section to the open weights file.

    filter_weights : list of per-filter word lists.  Each inner list has
                     n_beats elements, each element is 32 bytes (256 bits).
    n_tiles        : total number of tiles that will use this batch
                     (= n_spatial_positions * n_filters_in_batch).
    n_beats        : beats per filter (2, 3, or 5).
    """
    n_filters    = len(filter_weights)
    n_words      = n_filters * n_beats   # total logical words in this batch
    # Always write a full SRAM load (32 logical words = 256 rows) so that
    # w_ptr never reads uninitialized X values when n_words < 32.
    SRAM_LOGICAL_DEPTH = 32
    wt_file.write(f"LOAD {SRAM_LOGICAL_DEPTH}\n")
    for f_words in filter_weights:
        for word_bytes in f_words:
            wt_file.write(word_to_hex(word_bytes) + "\n")
    # Zero-pad remaining SRAM rows.
    zero_word = "00" * 32
    for _ in range(SRAM_LOGICAL_DEPTH - n_words):
        wt_file.write(zero_word + "\n")


# ---------------------------------------------------------------------------
# Write stimulus batch to file
# ---------------------------------------------------------------------------
def write_stimulus_batch(stim_file,
                         tiles: list[tuple[int, int, list[bytes]]]) -> None:
    """
    Write a stimulus batch.  Each tile is (cfg_beats, w_base_logical, beats)
    where beats is a list of 32-byte words.

    Format: "# BATCH <n_tiles>" followed by one line per tile:
      "<cfg_beats> <w_base_logical> <beat0_hex> [<beat1_hex> ...]"
    """
    stim_file.write(f"# BATCH {len(tiles)}\n")
    for (cfg_beats, w_base, beats) in tiles:
        hex_parts = [word_to_hex(b) for b in beats]
        stim_file.write(f"{cfg_beats} {w_base} " + " ".join(hex_parts) + "\n")


# ---------------------------------------------------------------------------
# Run iverilog simulation for one full set of weight+stimulus files
# ---------------------------------------------------------------------------
def run_simulation(run_dir: Path) -> list[int]:
    """
    Run vvp in run_dir (so the TB reads/writes files relative to that dir).
    Returns list of integer dot products, one per tile line in results file.
    """
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


# ---------------------------------------------------------------------------
# Compile the iverilog simulation binary (once at startup)
# ---------------------------------------------------------------------------
def compile_sim() -> None:
    """Compile all RTL sources with iverilog.  Raises if any source is missing."""
    print("[COMPILE] Checking RTL sources...")
    missing = [str(s) for s in RTL_SRCS if not s.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing RTL source(s):\n  " + "\n  ".join(missing)
        )
    cmd = (
        ["iverilog", "-g2012", "-o", str(SIM_BINARY)]
        + [str(s) for s in RTL_SRCS]
    )
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
    layer_name:  str,
    params:      dict,
    act_map_bin: np.ndarray,  # [in_ch, H, W] int8 {+1,-1}
    n_beats:     int,
    stride:      int,
    pad:         int,
    filters_per_batch: int,
    run_dir:     Path,
) -> np.ndarray:
    """
    Run one BNN conv layer through the RTL.

    Returns output_map: np.ndarray int8 {+1,-1} of shape [out_ch, out_H, out_W].
    """
    weights_bin = params["weights_bin"]   # [out_ch, in_ch, kH, kW]
    out_ch, in_ch, kH, kW = weights_bin.shape
    vec_len = in_ch * kH * kW             # bits per filter

    # Derive output spatial dimensions
    in_H, in_W = act_map_bin.shape[1], act_map_bin.shape[2]
    out_H = (in_H + 2 * pad - kH) // stride + 1
    out_W = (in_W + 2 * pad - kW) // stride + 1

    n_spatial = out_H * out_W

    # Compute per-filter 256-bit words (packed beats)
    # Each filter: flatten [in_ch, kH, kW] → 1-D binary vector → pack to beats
    all_filter_words = []  # list of list-of-bytes, length out_ch
    for f in range(out_ch):
        fvec  = weights_bin[f].reshape(-1)  # [in_ch*kH*kW] int8 {+1,-1}
        words = pack_to_256bit_words(fvec)  # list of 32-byte words
        # Should have exactly n_beats words (Python pads to multiple of 256)
        assert len(words) == n_beats, (
            f"{layer_name} filter {f}: expected {n_beats} beats, got {len(words)}"
        )
        all_filter_words.append(words)

    # Allocate output dot-product map [out_ch, out_H, out_W]
    dot_map = np.zeros((out_ch, out_H, out_W), dtype=np.int32)

    # Pre-extract all spatial receptive fields as packed beats.
    # Shape: [out_H, out_W, n_beats * 32 bytes]
    print(f"    [{layer_name}] extracting {n_spatial} receptive fields ...")
    rf_beats = []  # list of list-of-bytes per spatial position
    for row in range(out_H):
        for col in range(out_W):
            rf_vec   = extract_receptive_field(act_map_bin, row, col,
                                               stride, pad, kH, kW)
            rf_words = pack_to_256bit_words(rf_vec)
            assert len(rf_words) == n_beats
            rf_beats.append(rf_words)   # rf_beats[sp_idx] = list of n_beats words

    # Process filters in batches of filters_per_batch
    n_batches  = math.ceil(out_ch / filters_per_batch)
    all_dots   = []  # will have out_ch * n_spatial entries

    for batch_idx in range(n_batches):
        f_start = batch_idx * filters_per_batch
        f_end   = min(f_start + filters_per_batch, out_ch)
        batch_filters = list(range(f_start, f_end))
        n_f_this = len(batch_filters)
        n_tiles_this = n_f_this * n_spatial

        print(f"    [{layer_name}] batch {batch_idx+1}/{n_batches}: "
              f"filters {f_start}–{f_end-1} ({n_tiles_this} tiles) ...", end="", flush=True)

        # Build weight and stimulus files
        wt_path   = run_dir / "hw_inference_weights.txt"
        stim_path = run_dir / "hw_inference_stimulus.txt"

        with open(wt_path, "w") as wt_file:
            write_weight_batch(
                wt_file,
                [all_filter_words[f] for f in batch_filters],
                n_tiles_this,
                n_beats,
            )

        tiles = []
        for sp_idx in range(n_spatial):
            for fj, f_abs in enumerate(batch_filters):
                # w_base_logical = fj * n_beats (filter fj in the batch)
                w_base = fj * n_beats
                tiles.append((n_beats, w_base, rf_beats[sp_idx]))

        with open(stim_path, "w") as stim_file:
            write_stimulus_batch(stim_file, tiles)

        # Run simulation
        batch_dots = run_simulation(run_dir)

        if len(batch_dots) != n_tiles_this:
            raise RuntimeError(
                f"{layer_name} batch {batch_idx}: expected {n_tiles_this} "
                f"results, got {len(batch_dots)}"
            )

        all_dots.extend(batch_dots)
        print(f" done")

    # Reshape all_dots from (sp0,f0), (sp0,f1), ..., (sp0,fN), (sp1,f0), ...
    # into [out_ch, out_H, out_W]
    dot_arr = np.array(all_dots, dtype=np.int32)
    # all_dots ordering: for each spatial position sp, then for each filter f
    # → shape [n_spatial, out_ch]
    dot_arr = dot_arr.reshape(n_spatial, out_ch)         # [sp, out_ch]
    dot_map_flat = dot_arr.T                              # [out_ch, sp]
    dot_map = dot_map_flat.reshape(out_ch, out_H, out_W) # [out_ch, H, W]

    # Post-process: alpha scale + BatchNorm + binarize
    alpha   = params["alpha"][:, None, None]     # [out_ch, 1, 1]
    mean    = params["bn_mean"][:, None, None]
    var     = params["bn_var"][:, None, None]
    gamma   = params["bn_weight"][:, None, None]
    beta    = params["bn_bias"][:, None, None]
    eps     = 1e-5

    scaled  = alpha * dot_map.astype(np.float32)
    y       = (scaled - mean) / np.sqrt(var + eps) * gamma + beta
    out_bin = np.where(y > 0, np.int8(1), np.int8(-1))  # {+1,-1}

    return out_bin   # [out_ch, out_H, out_W]


# ---------------------------------------------------------------------------
# Full hardware inference for one image
# ---------------------------------------------------------------------------
def hw_inference_one_image(
    img_path:    str,
    model:       BNNClassifier,
    layer_params: dict,
    run_dir:     Path,
) -> tuple[int, float]:
    """
    Run one image through the hardware co-simulation path.

    Returns (predicted_class, confidence_pct) — class 0 = blank, 1 = non_blank.
    confidence is computed from the fc layer logits.
    """
    img_tensor = load_image_tensor(img_path)   # [1,3,224,224]

    # ── conv1 + bn1 on host (float) ──────────────────────────────────────────
    with torch.no_grad():
        x = model.bn1(model.conv1(img_tensor))  # [1,32,224,224]

    # Binarize to get binary activation map for conv2 input
    act2_np = x[0].numpy()   # [32, 224, 224] float
    act2_bin = binarize_np(act2_np)  # [32, 224, 224] int8 {+1,-1}

    # ── conv2 (BNN, stride=2, pad=1, kH=kW=3) → [64, 112, 112] ──────────────
    print(f"  [conv2] 64 output channels, 2 beats/filter, 16 filters/batch ...")
    act3_bin = hw_bnn_layer_forward(
        layer_name        = "conv2",
        params            = layer_params["conv2"],
        act_map_bin       = act2_bin,
        n_beats           = 2,
        stride            = 2,
        pad               = 1,
        filters_per_batch = 16,
        run_dir           = run_dir,
    )  # [64, 112, 112]

    # ── conv3 (BNN, stride=2, pad=1, kH=kW=3) → [128, 56, 56] ──────────────
    print(f"  [conv3] 128 output channels, 3 beats/filter, 10 filters/batch ...")
    act4_bin = hw_bnn_layer_forward(
        layer_name        = "conv3",
        params            = layer_params["conv3"],
        act_map_bin       = act3_bin,
        n_beats           = 3,
        stride            = 2,
        pad               = 1,
        filters_per_batch = 10,
        run_dir           = run_dir,
    )  # [128, 56, 56]

    # ── conv4 (BNN, stride=2, pad=1, kH=kW=3) → [256, 28, 28] ──────────────
    print(f"  [conv4] 256 output channels, 5 beats/filter, 6 filters/batch ...")
    act5_bin = hw_bnn_layer_forward(
        layer_name        = "conv4",
        params            = layer_params["conv4"],
        act_map_bin       = act4_bin,
        n_beats           = 5,
        stride            = 2,
        pad               = 1,
        filters_per_batch = 6,
        run_dir           = run_dir,
    )  # [256, 28, 28]

    # ── AdaptiveAvgPool + fc (host) ───────────────────────────────────────────
    # act5_bin is {+1,-1}; convert to float, pool, linear
    feat = torch.from_numpy(act5_bin.astype(np.float32)).unsqueeze(0)  # [1,256,28,28]
    with torch.no_grad():
        pooled  = model.pool(feat)                  # [1,256,1,1]
        flat    = torch.flatten(pooled, 1)          # [1,256]
        logits  = model.fc(flat)                    # [1,2]
        probs   = torch.softmax(logits, dim=1)[0]  # [2]

    pred       = int(probs.argmax().item())
    confidence = float(probs[pred].item()) * 100.0
    return pred, confidence


# ---------------------------------------------------------------------------
# Collect test images
# ---------------------------------------------------------------------------
def is_night_image(path: str) -> bool:
    """Return True if the image is a greyscale IR (night) capture.

    Night-vision cameras store IR frames as RGB with R==G==B on every pixel.
    Sample 200 pixels; if all are achromatic the image is night/IR.
    """
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


def collect_test_images(n_images: int | None = None,
                        single_image: str | None = None,
                        tod: str | None = None) -> list[str]:
    """Collect test images, optionally filtered by time-of-day.

    tod: 'day', 'night', or None (no filter).
    """
    if single_image:
        return [single_image]

    blank_dir    = TEST_DIR / "blank"
    nonblank_dir = TEST_DIR / "non_blank"

    if not blank_dir.exists() or not nonblank_dir.exists():
        raise FileNotFoundError(
            f"Test directories not found:\n  {blank_dir}\n  {nonblank_dir}"
        )

    blank_imgs    = sorted(blank_dir.glob("*.jpg")) + sorted(blank_dir.glob("*.png"))
    nonblank_imgs = sorted(nonblank_dir.glob("*.jpg")) + sorted(nonblank_dir.glob("*.png"))

    if tod is not None:
        want_night = (tod == "night")
        print(f"[IMAGES] Filtering for {tod} images — scanning {len(blank_imgs)+len(nonblank_imgs)} candidates ...")
        blank_imgs    = [p for p in blank_imgs    if is_night_image(str(p)) == want_night]
        nonblank_imgs = [p for p in nonblank_imgs if is_night_image(str(p)) == want_night]
        print(f"[IMAGES] After filter: {len(blank_imgs)} blank, {len(nonblank_imgs)} non_blank")

    if n_images is None:
        return [str(p) for p in blank_imgs + nonblank_imgs]

    half    = n_images // 2
    extra   = n_images - half
    blanks    = random.sample(blank_imgs,    min(half,  len(blank_imgs)))
    nonblanks = random.sample(nonblank_imgs, min(extra, len(nonblank_imgs)))
    imgs      = [str(p) for p in blanks + nonblanks]
    random.shuffle(imgs)
    return imgs


# ---------------------------------------------------------------------------
# Ground-truth label from path
# ---------------------------------------------------------------------------
def path_to_label(path: str) -> int:
    """0 = blank, 1 = non_blank (matches ImageFolder alphabetical sort)."""
    p = Path(path)
    if "non_blank" in p.parts or "non_blank" in p.parent.name:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BNN hardware inference co-simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--image",    metavar="PATH",
                     help="Single image to run through hardware")
    grp.add_argument("--n-images", metavar="N", type=int,
                     help="Pick N random test images (half blank, half non_blank)")
    parser.add_argument("--checkpoint", metavar="PATH",
                        default=str(DEFAULT_CHECKPOINT),
                        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--tod", choices=["day", "night"], default=None,
                        help="Filter test images to day (colour) or night (IR greyscale). "
                             "Use with day/night-specific checkpoints.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image selection (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not shutil.which("iverilog"):
        print("ERROR: iverilog not found on PATH.")
        print("  Install with: brew install icarus-verilog  (macOS)")
        print("              or: sudo apt install iverilog    (Debian/Ubuntu)")
        sys.exit(1)
    if not shutil.which("vvp"):
        print("ERROR: vvp not found on PATH (should be installed with iverilog).")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # ── Compile RTL (once) ────────────────────────────────────────────────────
    compile_sim()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[MODEL] Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint)
    print("[MODEL] Loaded.")

    # ── Extract layer parameters ──────────────────────────────────────────────
    print("[MODEL] Extracting layer parameters ...")
    layer_params = {
        "conv2": extract_layer_params(model, "conv2"),
        "conv3": extract_layer_params(model, "conv3"),
        "conv4": extract_layer_params(model, "conv4"),
    }
    for lname, p in layer_params.items():
        print(f"  {lname}: weights {p['weights_bin'].shape}, "
              f"alpha range [{p['alpha'].min():.3f}, {p['alpha'].max():.3f}]")

    # ── Collect images ────────────────────────────────────────────────────────
    images = collect_test_images(
        n_images     = args.n_images,
        single_image = args.image,
        tod          = args.tod,
    )
    print(f"\n[IMAGES] {len(images)} image(s) to process.")

    # ── Create a shared temp run directory ───────────────────────────────────
    run_dir = THIS_DIR / "hw_inference_run"
    run_dir.mkdir(exist_ok=True)

    # ── Per-image inference ───────────────────────────────────────────────────
    CLASS_NAMES = ["blank", "non_blank"]
    n_match = 0
    n_total = 0

    print()
    print("=" * 80)
    print(f"{'Image':<36}  {'ToD':>5}  {'GT':>8}  {'PyTorch':>8}  {'HW':>8}  {'Match':>6}")
    print("-" * 80)

    for img_path in images:
        n_total += 1
        img_name = Path(img_path).name
        gt_label = path_to_label(img_path)
        tod_label = "night" if is_night_image(img_path) else "day"
        gt_name  = CLASS_NAMES[gt_label]

        print(f"\n[IMAGE {n_total}/{len(images)}] {img_name}")

        # PyTorch reference
        img_tensor = load_image_tensor(img_path)
        pt_pred, pt_conf = pytorch_full_forward(model, img_tensor)
        pt_name = CLASS_NAMES[pt_pred]
        print(f"  PyTorch: {pt_name} ({pt_conf:.1f}%)")

        # Hardware co-simulation
        try:
            hw_pred, hw_conf = hw_inference_one_image(
                img_path    = img_path,
                model       = model,
                layer_params = layer_params,
                run_dir     = run_dir,
            )
        except Exception as exc:
            print(f"  HW ERROR: {exc}")
            print(f"  {'<error>':<36}  {tod_label:>5}  {gt_name:>8}  {pt_name:>8}  {'ERR':>8}  {'N':>6}")
            continue

        hw_name = CLASS_NAMES[hw_pred]
        match   = (hw_pred == pt_pred)
        if match:
            n_match += 1
        match_str = "Y" if match else "N"

        print(f"  HW:      {hw_name} ({hw_conf:.1f}%)")
        print(f"  Match:   {match_str}")

        # Table row
        print(f"  {img_name:<36}  {tod_label:>5}  {gt_name:>8}  {pt_name:>8}  {hw_name:>8}  {match_str:>6}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"SUMMARY")
    print(f"  Images processed : {n_total}")
    print(f"  HW matches PyTorch: {n_match}/{n_total}  ({100.0*n_match/n_total if n_total else 0:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
