# Milestone 2 — Reproducibility

**Project:** Ultra-Low-Power BNN Trail Cam Smart Filter
**Course:** ECE 510 — Hardware for AI, Spring 2026
**Author:** Rebecca Gilbert-Croysdale

---

## Tool Versions

| Tool                            | Version | Install                                   |
| ------------------------------- | ------- | ----------------------------------------- |
| **Icarus Verilog** (`iverilog`) | 13.0    | `brew install icarus-verilog`             |
| **VVP simulator**               | 13.0    | (bundled with Icarus)                     |
| **GTKWave**                     | 3.4.0   | `brew install gtkwave` (waveform viewing) |
| **Python**                      | 3.14.3  | (precision validation only)               |
| **PyTorch**                     | 2.11.0  | (precision validation only)               |

All HDL commands run from the **repository root** (`HW-4-AI-ML/`).

---

## 1. Compute Core Testbench

**Files:**

- RTL: `project/m2/rtl/compute_core.sv` — module `compute_core`
- TB: `project/m2/tb/tb_compute_core.sv` — module `tb_compute_core`

```bash
mkdir -p project/m2/sim

# Compile
iverilog -g2012 -Wall \
  -o project/m2/sim/compute_core.vvp \
  project/m2/rtl/compute_core.sv \
  project/m2/tb/tb_compute_core.sv

# Run and save log
vvp project/m2/sim/compute_core.vvp | tee project/m2/sim/compute_core_run.log
```

Expected last lines of `compute_core_run.log`:

```
VERIFIABLE PASS — all vectors matched $countones reference
```

**How verification works:** The DUT uses a combinational `for` loop to count XNOR bits. The testbench uses SystemVerilog's built-in `$countones()` — a fully independent reference — to compute the expected accumulator value and asserts equality after every valid handshake.

---

## 2. Interface Testbench

**Files:**

- RTL: `project/m2/rtl/interface.sv` — module `axis_interface`
- TB: `project/m2/tb/tb_interface.sv` — module `tb_interface`

> **Note on module name:** `interface` is a reserved SystemVerilog keyword; naming
> a module `interface` causes a fatal compilation error in all standards-compliant
> tools. The module is therefore named `axis_interface`. The filename is
> `interface.sv` exactly as the rubric specifies.

```bash
# Compile
iverilog -g2012 -Wall \
  -o project/m2/sim/interface.vvp \
  project/m2/rtl/interface.sv \
  project/m2/tb/tb_interface.sv

# Run and save log
vvp project/m2/sim/interface.vvp | tee project/m2/sim/interface_run.log
```

Expected last lines of `interface_run.log`:

```
VERIFIABLE PASS — All AXI protocols and data matched
```

**How verification works:** A SV queue acts as the reference model. A concurrent monitor pushes every accepted beat (tvalid && tready) into the queue on the input side and pops/compares on the output side (core_valid && core_ready). Phase 1 tests 50 transactions at full throughput; Phase 2 tests 50 transactions with random `core_ready` backpressure to exercise the skid buffer.

---

## 3. Waveform

`project/m2/sim/waveform.png` — screenshot from GTKWave showing the compute_core accumulator behavior: reset, 4 accumulation cycles, accum_clear.

To generate a VCD for GTKWave, add the following to `tb_compute_core.sv` inside `initial begin`:

```systemverilog
$dumpfile("project/m2/sim/compute_core.vcd");
$dumpvars(0, tb_compute_core);
```

Then recompile and run; open `compute_core.vcd` in GTKWave.

---

## 4. Precision Validation

```bash
python3 project/m2/validate_precision.py
```

Requires PyTorch 2.11+, torchvision, and the checkpoint `project/bnn_distilled_876pct.pth`.
Results are documented in `project/m2/precision.md`.

---

## 5. Deviations from M1 Plan

`codefest/cf02/analysis/partition_rationale.md` planned to move **"all three
`BinarizeConv2d` layers"** to the custom accelerator and keep **"data loading, raw
input binarization, BatchNorm2d, pooling, and the final linear classifier"** on the
host CPU.  Two changes were made before M2 RTL:

| Item | M1 Plan (`partition_rationale.md`) | M2 Reality | Reason |
|------|-------------------------------------|------------|--------|
| **Layer count** | 3 `BinarizeConv2d` layers on chiplet (Conv1–3) | **4 layers total**: Conv2–4 (1-bit XNOR) on chiplet; Conv1 is a new 4th layer added *above* the original 3 (128→256 channels on Conv4). | Adding Conv4 (128→256 channels) widened the final feature map before global pooling, raising val accuracy 73.4% → 76.2% with no interface or precision changes. |
| **Conv1 type and partition** | Conv1 was one of the three `BinarizeConv2d` layers — binarized and mapped to the chiplet. | Conv1 is now a standard `nn.Conv2d` (not binarized), running as **INT8 fixed-point on the ARM host CPU**. Only Conv2–4 (`BinarizeConv2d`) remain on the chiplet. | Applying `sign()` binarization to raw 8-bit RGB pixel inputs collapsed accuracy. Replacing Conv1 with an INT8 standard conv on the host gave 85.2% vs 76.2% for the all-binary variant (+9%). This is consistent with the M1 plan's note that "raw input binarization" stays in software — Conv1 is the layer that performs that binarization boundary. Validated by fake-quantization: −4.0% accuracy delta vs FP32, logit MAE 0.2453 (see `precision.md`). |
| **Interface protocol** | AXI4-Stream, 256-bit @ 300 MHz (`interface_selection.md`: 9.6 GB/s rated, 8.0 GB/s required) | **Unchanged.** | The INT8 Conv1 output payload (1.6 MB/frame) is 4× smaller than the equivalent FP32 payload (6.4 MB), adding margin on top of the original 20% bandwidth headroom. The 9.6 GB/s interface is more than sufficient. |
| **Chiplet compute precision** | 1-bit XNOR-Popcount (AI: 394.8 FLOP/byte, target design point 150 FLOP/byte @ 1200 GFLOP/s) | **Unchanged.** | M1 roofline analysis confirmed the chiplet operates deeply compute-bound. Moving Conv1 off-chip does not affect this operating point. |
