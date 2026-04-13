"""
baseline_profile.py
===================
Software baseline for BNN "Animal vs. Empty" trail-camera classifier.
ECE 510 Spring 2026 — M1 deliverable.

Run:
    python3 baseline_profile.py

Outputs (to stdout + baseline_profile_output.txt):
  - Hardware platform spec
  - torchinfo model summary (MACs, parameters per layer)
  - cProfile timing over 100 forward passes
  - Arithmetic intensity calculation for the dominant conv layer

All numbers printed here become your M4 comparison baseline.
Document the platform section carefully — the benchmark is only
reproducible if someone else uses identical hardware.
"""

# ── macOS OpenMP workaround (must happen before torch import) ──────────────
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import platform
import cProfile
import pstats
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary


# ===========================================================================
# 0. REDIRECT OUTPUT TO BOTH STDOUT AND A FILE
#    Every number you need for M1/M4 is captured here.
# ===========================================================================
class Tee:
    """Writes to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile  = open(filepath, "w")
    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = Tee("baseline_profile_output.txt")


# ===========================================================================
# 1. HARDWARE PLATFORM LOGGING
#    Syllabus requirement: "document the platform (CPU or GPU, hardware spec)
#    so the M4 comparison is reproducible."
# ===========================================================================
print("=" * 70)
print("PLATFORM SPECIFICATION")
print("=" * 70)
print(f"OS              : {platform.system()} {platform.release()} ({platform.version()})")
print(f"Machine         : {platform.machine()}")
print(f"Processor       : {platform.processor()}")
print(f"Python          : {platform.python_version()}")
print(f"PyTorch         : {torch.__version__}")

# CPU core count
try:
    import psutil
    cores_physical = psutil.cpu_count(logical=False)
    cores_logical  = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"CPU cores       : {cores_physical} physical / {cores_logical} logical")
    print(f"System RAM      : {ram_gb:.1f} GB")
except ImportError:
    import multiprocessing
    print(f"CPU cores       : {multiprocessing.cpu_count()} logical")

# GPU (MPS on Apple Silicon, CUDA on NVIDIA)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Accelerator     : Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Accelerator     : CUDA — {torch.cuda.get_device_name(0)}")
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print(f"Accelerator     : None (CPU only)")

# IMPORTANT: baseline runs on CPU to match a "general-purpose processor" baseline.
# The hardware accelerator will be compared against this CPU number.
device = torch.device("cpu")
print(f"\nBaseline device : CPU  (intentional — GPU/MPS not used for the baseline)")
print(f"Batch size      : 1    (single-image inference, as used throughout M1–M4)")
print()


# ===========================================================================
# 2. MOCK DATASET
#    224×224 RGB images, binary labels (0=empty, 1=animal).
#    Using torch.randn so no download is needed and results are reproducible
#    given the fixed seed below.
#    Batch size = 1 is documented above and used throughout.
# ===========================================================================
BATCH_SIZE   = 1
NUM_CLASSES  = 2       # "animal" vs "empty"
NUM_SAMPLES  = 100     # enough for stable timing; matches cProfile run count
IMG_CHANNELS = 3
IMG_H, IMG_W = 224, 224
SEED         = 42

torch.manual_seed(SEED)

class MockTrailCamDataset(Dataset):
    """
    Returns (image, label) pairs where:
      image  : torch.Tensor of shape (3, 224, 224), dtype float32
      label  : int in {0, 1}
    Data is pre-generated in __init__ so DataLoader iteration is
    not a timing bottleneck during profiling.
    """
    def __init__(self, num_samples: int):
        self.images = torch.randn(num_samples, IMG_CHANNELS, IMG_H, IMG_W)
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

dataset    = MockTrailCamDataset(NUM_SAMPLES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("=" * 70)
print("DATASET SPECIFICATION")
print("=" * 70)
print(f"Dataset type    : Mock (torch.randn, seed={SEED})")
print(f"Image size      : {IMG_CHANNELS}×{IMG_H}×{IMG_W} (RGB, 224×224)")
print(f"Num samples     : {NUM_SAMPLES}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Classes         : {NUM_CLASSES}  (0=empty, 1=animal)")
print()


# ===========================================================================
# 3. BNN MODEL DEFINITION
#
# BinarizeConv2d: during the forward pass, weights are binarized to +1/-1
# using torch.sign() before the convolution. This simulates the 1-bit weight
# constraint of a BNN in software.  Inputs are also binarized via sign().
#
# In real hardware, these +1/-1 MACs collapse to XNOR+Popcount operations.
# In software (this baseline), they still run as float32 multiplications —
# that is the inefficiency the hardware accelerator eliminates.
#
# Architecture:
#   BinarizeConv2d(3,  32, 3×3, stride=1, pad=1) → BN → sign → 32×224×224
#   BinarizeConv2d(32, 64, 3×3, stride=2, pad=1) → BN → sign → 64×112×112
#   BinarizeConv2d(64,128, 3×3, stride=2, pad=1) → BN → sign → 128×56×56
#   AdaptiveAvgPool2d(1×1)                                   → 128×1×1
#   Flatten                                                   → 128
#   Linear(128, 2)                                            → logits
#
# Why this size? 3 conv layers give a meaningful MAC count (visible on
# the roofline) while keeping the network small enough to run 100×
# inference quickly on a CPU.
# ===========================================================================

class BinarizeConv2d(nn.Conv2d):
    """
    Conv2d where weights (and inputs) are binarized to {+1, -1}
    using torch.sign() on every forward pass.

    Training note: In a real trained BNN you would use the Straight-Through
    Estimator (STE) for gradients. We skip training here — this baseline
    measures inference performance only.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binarize weights: +1 if weight >= 0, else -1
        binary_weights = torch.sign(self.weight)
        # Replace zeros (sign(0)=0) with +1 to stay in {+1,-1}
        binary_weights = torch.where(binary_weights == 0,
                                     torch.ones_like(binary_weights),
                                     binary_weights)

        # Binarize input activations
        binary_input = torch.sign(x)
        binary_input = torch.where(binary_input == 0,
                                   torch.ones_like(binary_input),
                                   binary_input)

        return F.conv2d(binary_input, binary_weights,
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class BNNClassifier(nn.Module):
    """
    Three-layer BNN for binary image classification.
    Dominant kernel: first BinarizeConv2d (largest spatial output,
    highest memory traffic).
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Layer 1: 3→32, stride 1 — keeps full 224×224 spatial size
        self.conv1 = BinarizeConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        # Layer 2: 32→64, stride 2 — downsamples to 112×112
        self.conv2 = BinarizeConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        # Layer 3: 64→128, stride 2 — downsamples to 56×56
        self.conv3 = BinarizeConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sign(self.bn1(self.conv1(x)))  # binarize post-BN
        x = torch.sign(self.bn2(self.conv2(x)))
        x = torch.sign(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


model = BNNClassifier(num_classes=NUM_CLASSES).to(device)
model.eval()  # inference mode — disables dropout, uses BN running stats


# ===========================================================================
# 4a. TORCHINFO MODEL SUMMARY
#     Prints MACs (multiply-accumulate ops) and parameter counts per layer.
#     This is your primary source for the FLOPs number in the
#     arithmetic intensity calculation.
# ===========================================================================
print("=" * 70)
print("TORCHINFO MODEL SUMMARY  (MACs and parameter sizes per layer)")
print("=" * 70)
model_stats = summary(
    model,
    input_size=(BATCH_SIZE, IMG_CHANNELS, IMG_H, IMG_W),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    verbose=1,         # prints the table
    device=device,
)
# torchinfo stores totals — save for arithmetic intensity section
total_macs    = model_stats.total_mult_adds   # total MACs across all layers
total_params  = model_stats.total_params
total_flops   = total_macs * 2                # 1 MAC = 1 multiply + 1 add = 2 FLOPs
print(f"\nSummary totals:")
print(f"  Total MACs   : {total_macs:,}")
print(f"  Total FLOPs  : {total_flops:,}  (MACs × 2)")
print(f"  Total params : {total_params:,}")
print()


# ===========================================================================
# 4b. cPROFILE — LATENCY AND THROUGHPUT
#     Run 100 single-image forward passes inside cProfile.
#     We also measure wall-clock time separately for clean throughput numbers.
# ===========================================================================
print("=" * 70)
print("cPROFILE — 100-IMAGE INFERENCE BENCHMARK")
print("=" * 70)

# Pre-generate a fixed input tensor (avoids DataLoader overhead in timing)
dummy_input = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_H, IMG_W).to(device)

def run_inference(n: int = 100):
    """Forward-pass loop used by both wall-clock timer and cProfile."""
    with torch.no_grad():
        for _ in range(n):
            _ = model(dummy_input)

N_RUNS = 100

# ── Wall-clock timing (simple, reproducible) ──────────────────────────────
# Warm-up: one pass to load any lazy-initialized state
with torch.no_grad():
    _ = model(dummy_input)

t_start = time.perf_counter()
run_inference(N_RUNS)
t_end   = time.perf_counter()

total_wall_s    = t_end - t_start
latency_ms      = (total_wall_s / N_RUNS) * 1000   # ms per image
throughput_fps  = N_RUNS / total_wall_s             # images per second

print(f"\nWall-clock results ({N_RUNS} runs, batch size = {BATCH_SIZE}):")
print(f"  Total time      : {total_wall_s*1000:.1f} ms")
print(f"  Avg latency     : {latency_ms:.3f} ms / image")
print(f"  Throughput      : {throughput_fps:.1f} samples/sec")
print()

# ── cProfile (call-level breakdown) ───────────────────────────────────────
profiler = cProfile.Profile()
profiler.enable()
run_inference(N_RUNS)
profiler.disable()

stream = io.StringIO()
ps = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
ps.print_stats(20)   # top 20 functions by cumulative time
print("cProfile top-20 functions (cumulative time):")
print(stream.getvalue())


# ===========================================================================
# 5. ARITHMETIC INTENSITY CALCULATION
#    Target kernel: conv1 — BinarizeConv2d(3, 32, 3×3, stride=1, pad=1)
#    Input spatial: 224×224
#    Output spatial: 224×224 (stride=1, same-padding)
#
#    We use the SOFTWARE floating-point representation (4 bytes per value)
#    because this is the CPU baseline. The hardware design uses 1-bit values,
#    which will radically change the memory-traffic side of the equation —
#    that is exactly the accelerator advantage you will quantify at M4.
#
#    "No data reuse" assumption: every weight, input activation, and output
#    activation is read/written once from/to main memory. This is the
#    conservative (pessimistic) bound and is standard for roofline analysis.
# ===========================================================================
print("=" * 70)
print("ARITHMETIC INTENSITY — conv1 (dominant kernel)")
print("=" * 70)

# ── conv1 dimensions ──────────────────────────────────────────────────────
C_in   = 3     # input channels  (RGB)
C_out  = 32    # output channels
K      = 3     # kernel size (3×3)
H_in   = 224   # input height
W_in   = 224   # input width
stride = 1
pad    = 1

H_out = (H_in + 2*pad - K) // stride + 1   # = 224
W_out = (H_in + 2*pad - K) // stride + 1   # = 224

# ── FLOPs ─────────────────────────────────────────────────────────────────
# For a conv layer:
#   MACs = C_out × H_out × W_out × C_in × K × K
#   FLOPs = 2 × MACs  (one multiply + one add per MAC)
MACs_conv1  = C_out * H_out * W_out * C_in * K * K
FLOPs_conv1 = 2 * MACs_conv1

# ── Memory traffic (bytes, float32 = 4 bytes, NO data reuse) ──────────────
BYTES_PER_FLOAT = 4

bytes_weights = C_out * C_in * K * K * BYTES_PER_FLOAT   # weight tensor
bytes_input   = C_in  * H_in * W_in  * BYTES_PER_FLOAT   # input activation
bytes_output  = C_out * H_out * W_out * BYTES_PER_FLOAT  # output activation

total_bytes = bytes_weights + bytes_input + bytes_output

# ── Arithmetic Intensity ──────────────────────────────────────────────────
AI = FLOPs_conv1 / total_bytes   # FLOPs per byte

print(f"\nconv1 configuration:")
print(f"  Input  : {C_in}×{H_in}×{W_in}")
print(f"  Kernel : {C_out} filters, {K}×{K}, stride={stride}, pad={pad}")
print(f"  Output : {C_out}×{H_out}×{W_out}")

print(f"\nFLOPs:")
print(f"  MACs            = C_out × H_out × W_out × C_in × K²")
print(f"                  = {C_out} × {H_out} × {W_out} × {C_in} × {K}²")
print(f"                  = {MACs_conv1:,}")
print(f"  FLOPs           = 2 × MACs = {FLOPs_conv1:,}")

print(f"\nMemory traffic (float32, no data reuse):")
print(f"  Weights         = {C_out}×{C_in}×{K}×{K} × 4B = {bytes_weights:,} B  ({bytes_weights/1024:.1f} KB)")
print(f"  Input acts      = {C_in}×{H_in}×{W_in} × 4B   = {bytes_input:,} B  ({bytes_input/1024:.1f} KB)")
print(f"  Output acts     = {C_out}×{H_out}×{W_out} × 4B = {bytes_output:,} B  ({bytes_output/1024:.1f} KB)")
print(f"  Total           = {total_bytes:,} B  ({total_bytes/1024/1024:.2f} MB)")

print(f"\nArithmetic Intensity (conv1):")
print(f"  AI = FLOPs / bytes = {FLOPs_conv1:,} / {total_bytes:,}")
print(f"  AI = {AI:.2f} FLOPs/byte")

# ── Full-network AI (for reference) ───────────────────────────────────────
# Use torchinfo's total MACs for a rough whole-model AI.
# Memory = all parameters + one input tensor + one output tensor (rough estimate).
bytes_all_params = total_params * BYTES_PER_FLOAT
bytes_model_input  = C_in * H_in * W_in * BYTES_PER_FLOAT
bytes_model_output = NUM_CLASSES * BYTES_PER_FLOAT
total_bytes_model  = bytes_all_params + bytes_model_input + bytes_model_output
AI_model = total_flops / total_bytes_model

print(f"\nWhole-model AI (rough, params-only memory estimate):")
print(f"  Total FLOPs     : {total_flops:,}")
print(f"  Total param mem : {bytes_all_params/1024/1024:.2f} MB")
print(f"  AI (model)      : {AI_model:.2f} FLOPs/byte")

print()
print("=" * 70)
print("HOW TO EXTRACT ROOFLINE NUMBERS FROM THIS OUTPUT")
print("=" * 70)
print("""
To plot your roofline model, you need two numbers per operating point:

  X-axis  →  Arithmetic Intensity (FLOPs/byte)
               Use the conv1 AI printed above for the dominant kernel.
               Use the whole-model AI for the full-network point.

  Y-axis  →  Attained Performance (FLOPs/sec)
               = Total FLOPs (conv1 or whole model)
                 ÷ Avg latency in seconds
               Compute from the wall-clock throughput section:
                 attained = total_flops / (latency_ms / 1000)

Roofline ceilings for a typical modern laptop CPU:
  Memory bandwidth ridge : ~40-60 GB/s  (check your spec sheet)
  Peak compute           : ~100-400 GFLOPs/s (FP32, all cores, AVX)

If your kernel's AI falls LEFT of the ridge point → memory-bound.
If it falls RIGHT                                 → compute-bound.
BNN conv layers typically land near the ridge because the weight
tensor is tiny (1-bit in hardware), shifting the bottleneck to compute.

Recommended tool: the Excel/Python roofline template from the course,
or use matplotlib with two lines:
  memory_roof(AI) = bandwidth * AI
  compute_roof    = peak_flops
  attained = min(memory_roof, compute_roof)  <- where your kernel sits
""")

# ── Restore stdout ─────────────────────────────────────────────────────────
sys.stdout.logfile.close()
sys.stdout = sys.stdout.terminal
print("\nDone. Full output saved to baseline_profile_output.txt")
