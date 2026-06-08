"""
serengeti2_profile.py
=====================
Benchmark the *actual* trained bnn_serengeti2 model (4-layer, real weights)
on the current machine.  Mirrors the methodology of baseline_profile.py so
numbers are directly comparable for the design-justification report.

Run:
    cd project/
    python serengeti2_profile.py

Outputs:
  - Platform spec
  - torchinfo model summary (MACs / FLOPs)
  - Wall-clock FPS over 100 forward passes (1 warm-up discarded)
  - Energy-per-frame estimate (requires manual power reading — see note)
  - Arithmetic intensity for conv2 (first binary BNN layer, dominant kernel)
  - Results written to serengeti2_profile_output.txt
"""

import os, sys, time, platform, io, cProfile, pstats
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from pathlib import Path

# ── Tee stdout to file ─────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.logfile  = open(path, "w")
    def write(self, msg):
        self.terminal.write(msg); self.logfile.write(msg)
    def flush(self):
        self.terminal.flush(); self.logfile.flush()

sys.stdout = Tee("serengeti2_profile_output.txt")

# ── Platform ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("PLATFORM SPECIFICATION")
print("=" * 70)
print(f"OS              : {platform.system()} {platform.release()} ({platform.version()})")
print(f"Machine         : {platform.machine()}")
print(f"Processor       : {platform.processor()}")
print(f"Python          : {platform.python_version()}")
print(f"PyTorch         : {torch.__version__}")

try:
    import psutil
    print(f"CPU cores       : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical")
    print(f"System RAM      : {psutil.virtual_memory().total / (1024**3):.1f} GB")
except ImportError:
    import multiprocessing
    print(f"CPU cores       : {multiprocessing.cpu_count()} logical")

if torch.backends.mps.is_available():
    print(f"Accelerator     : Apple MPS (Metal Performance Shaders) — available but NOT used")
elif torch.cuda.is_available():
    print(f"Accelerator     : CUDA ({torch.cuda.get_device_name(0)}) — available but NOT used")
else:
    print(f"Accelerator     : None")

DEVICE = torch.device("cpu")
print(f"\nBenchmark device: CPU  (matches M1 baseline methodology)")
print(f"Batch size      : 1    (single-image inference)")
print()


# ── Model definition (must match bnn_serengeti2.py exactly) ───────────────────
class _STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        b = torch.sign(x)
        return torch.where(b == 0, torch.ones_like(x), b)
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1.0).float()

binarize = _STESign.apply


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.ones(self.out_channels))

    def forward(self, x):
        binary_w = binarize(self.weight)
        binary_x = binarize(x)
        out = F.conv2d(binary_x, binary_w, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)
        return out * self.alpha.view(1, -1, 1, 1)


class BNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = BinarizeConv2d(32,  64,  3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = BinarizeConv2d(64,  128, 3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = BinarizeConv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = binarize(self.bn2(self.conv2(x)))
        x = binarize(self.bn3(self.conv3(x)))
        x = binarize(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── Load trained weights ───────────────────────────────────────────────────────
CHECKPOINT = Path(__file__).parent / "bnn_serengeti2.pth"
model = BNNClassifier(num_classes=2).to(DEVICE)

print("=" * 70)
print("MODEL WEIGHTS")
print("=" * 70)
if CHECKPOINT.exists():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    # checkpoint may be a raw state_dict or wrapped dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        print(f"Checkpoint keys : {[k for k in ckpt if k != 'model']}")
        if "epoch" in ckpt:
            print(f"Saved at epoch  : {ckpt['epoch']}")
        if "best_val_acc" in ckpt:
            print(f"Best val acc    : {ckpt['best_val_acc']:.4f}")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING missing keys : {missing}")
    if unexpected:
        print(f"WARNING unexpected   : {unexpected}")
    print(f"Checkpoint      : {CHECKPOINT.name}  (LOADED)")
else:
    print(f"WARNING: {CHECKPOINT} not found — using random weights")

model.eval()
print()


# ── torchinfo summary ──────────────────────────────────────────────────────────
print("=" * 70)
print("TORCHINFO MODEL SUMMARY")
print("=" * 70)
IMG_C, IMG_H, IMG_W = 3, 224, 224
model_stats = summary(
    model,
    input_size=(1, IMG_C, IMG_H, IMG_W),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    verbose=1,
    device=DEVICE,
)
total_macs   = model_stats.total_mult_adds
total_params = model_stats.total_params
total_flops  = total_macs * 2
print(f"\nSummary totals:")
print(f"  Total MACs   : {total_macs:,}")
print(f"  Total FLOPs  : {total_flops:,}  (MACs × 2)")
print(f"  Total params : {total_params:,}")
print()


# ── Timing benchmark ───────────────────────────────────────────────────────────
print("=" * 70)
print("WALL-CLOCK BENCHMARK  (100 runs, batch=1, CPU)")
print("=" * 70)

dummy = torch.randn(1, IMG_C, IMG_H, IMG_W).to(DEVICE)
N = 100

with torch.no_grad():
    _ = model(dummy)          # warm-up

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N):
        _ = model(dummy)
t1 = time.perf_counter()

total_s    = t1 - t0
latency_ms = total_s / N * 1000
fps        = N / total_s

print(f"\nResults ({N} runs):")
print(f"  Total time      : {total_s*1000:.1f} ms")
print(f"  Avg latency     : {latency_ms:.3f} ms / image")
print(f"  Throughput      : {fps:.1f} FPS")
print()


# ── cProfile breakdown ────────────────────────────────────────────────────────
def _run(n=100):
    with torch.no_grad():
        for _ in range(n):
            _ = model(dummy)

prof = cProfile.Profile()
prof.enable(); _run(N); prof.disable()
buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(20)
print("cProfile top-20 functions (cumulative time):")
print(buf.getvalue())


# ── Arithmetic intensity — conv2 (first binary BNN layer) ─────────────────────
print("=" * 70)
print("ARITHMETIC INTENSITY — conv2 (first binary BNN layer)")
print("=" * 70)

C_in, C_out, K = 32, 64, 3
H_in, W_in     = 224, 224
stride, pad    = 2, 1
H_out = (H_in + 2*pad - K) // stride + 1   # 112
W_out = (W_in + 2*pad - K) // stride + 1   # 112

MACs_conv2  = C_out * H_out * W_out * C_in * K * K
FLOPs_conv2 = 2 * MACs_conv2

B = 4   # float32 bytes
bytes_w   = C_out * C_in * K * K * B
bytes_in  = C_in  * H_in * W_in  * B
bytes_out = C_out * H_out * W_out * B
total_bytes = bytes_w + bytes_in + bytes_out
AI_conv2 = FLOPs_conv2 / total_bytes

print(f"\nconv2: {C_in}×{H_in}×{W_in} → {C_out}×{H_out}×{W_out}  (stride={stride})")
print(f"  MACs          : {MACs_conv2:,}")
print(f"  FLOPs         : {FLOPs_conv2:,}")
print(f"  Memory (f32)  : {total_bytes:,} B  ({total_bytes/1024/1024:.2f} MB)")
print(f"  AI            : {AI_conv2:.2f} FLOPs/byte")
print()


# ── Energy-per-frame estimate ─────────────────────────────────────────────────
print("=" * 70)
print("ENERGY PER FRAME ESTIMATE")
print("=" * 70)
print("""
To get energy/frame on this machine:

  Option A — sudo powermetrics (macOS):
    In a separate terminal, start:
      sudo powermetrics --samplers cpu_power -i 500 -n 20
    Then immediately run this script.
    Read the 'CPU Power' mW values while inference is running and average them.
    Energy/frame = avg_mW × latency_ms / 1000  (in mJ)

  Option B — estimate from TDP:
    If you know your CPU's nominal TDP (W), energy/frame ≈ TDP × latency_s

The M1 baseline used ~10,000–15,000 mW (10–15 W) CPU package power.
""")
print(f"  Latency this run : {latency_ms:.3f} ms")
print(f"  At 10 W          : {10000 * latency_ms / 1000:.1f} mJ/frame")
print(f"  At 15 W          : {15000 * latency_ms / 1000:.1f} mJ/frame")
print()


# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY — paste into design_justification.md")
print("=" * 70)
print(f"  Model           : bnn_serengeti2 (4-layer, trained weights)")
print(f"  Platform        : {platform.processor() or platform.machine()}")
print(f"  Total FLOPs     : {total_flops:,}")
print(f"  Total params    : {total_params:,}")
print(f"  Latency         : {latency_ms:.3f} ms/image")
print(f"  Throughput      : {fps:.1f} FPS")
print(f"  AI (conv2, f32) : {AI_conv2:.2f} FLOPs/byte")
print()

sys.stdout.logfile.close()
sys.stdout = sys.stdout.terminal
print(f"\nDone. Full output saved to serengeti2_profile_output.txt")
