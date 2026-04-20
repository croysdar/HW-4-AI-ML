"""
Google Colab runner for cf03 CUDA GEMM kernels.
Paste each cell block (separated by # ── CELL N ──) into its own Colab cell.
Upload gemm_naive.cu and gemm_tiled.cu to Colab first (Files panel → Upload).
"""

# ── CELL 1: verify GPU and CUDA ──────────────────────────────────────────────
import subprocess, os

def run(cmd, **kw):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result

run("nvidia-smi")
run("nvcc --version")


# ── CELL 2: compile both kernels ─────────────────────────────────────────────
r1 = run("nvcc -O3 -o naive gemm_naive.cu -lm")
r2 = run("nvcc -O3 -o tiled gemm_tiled.cu -lm")

if r1.returncode == 0:
    print("gemm_naive compiled OK")
else:
    print("gemm_naive COMPILE ERROR — check output above")

if r2.returncode == 0:
    print("gemm_tiled compiled OK")
else:
    print("gemm_tiled COMPILE ERROR — check output above")


# ── CELL 3: run both kernels and print GFLOP/s ───────────────────────────────
print("=" * 50)
print("Running naive kernel:")
run("./naive")

print("=" * 50)
print("Running tiled kernel:")
run("./tiled")


# ── CELL 4: profile with Nsight Compute (ncu) ────────────────────────────────
# ncu may require root or relaxed ptrace scope in Colab; the --no-nvlink flag
# avoids a common Colab permission error. Remove it if your environment has
# full profiling access.
NCU_METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
)

ncu_naive_cmd = (
    f"ncu --metrics {NCU_METRICS} "
    "--target-processes all "
    "./naive 2>&1"
)
ncu_tiled_cmd = (
    f"ncu --metrics {NCU_METRICS} "
    "--target-processes all "
    "./tiled 2>&1"
)

print("Profiling naive kernel (this may take ~1–2 min)...")
naive_profile = subprocess.run(ncu_naive_cmd, shell=True, capture_output=False,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
naive_out = naive_profile.stdout
print(naive_out)

print("Profiling tiled kernel (this may take ~1–2 min)...")
tiled_profile = subprocess.run(ncu_tiled_cmd, shell=True, capture_output=False,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
tiled_out = tiled_profile.stdout
print(tiled_out)


# ── CELL 5: save all profiling output to ncu_output.txt ──────────────────────
output_path = "ncu_output.txt"

with open(output_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("NCU PROFILE — gemm_naive\n")
    f.write("=" * 60 + "\n")
    f.write(naive_out)
    f.write("\n\n")
    f.write("=" * 60 + "\n")
    f.write("NCU PROFILE — gemm_tiled\n")
    f.write("=" * 60 + "\n")
    f.write(tiled_out)

print(f"Saved to {output_path}")

# Download the file to your local machine (Colab-specific helper)
try:
    from google.colab import files
    files.download(output_path)
    print("Download triggered.")
except ImportError:
    print("Not in Colab — file saved locally.")
