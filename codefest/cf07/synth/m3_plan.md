# M3 Plan — compute_core

## Core Problem

The 14-level combinational path (4057 ps min-load) violates the 3330 ps budget. Pipelining
is mandatory.

## Change 1: Insert pipeline register after XNOR stage (required)

The XNOR stage is 1 level; the downstream 9-level xnor3/xor3 adder tree contributes 3262 ps.
Adding one register at the XNOR output creates two ~7-cell half-paths, each ~2.1–2.3 ns —
within the 3330 ps budget. Throughput stays at 1 tile/clock; latency increases by 1 cycle.

## Change 2: Eliminate 30 unexpected lpflow power cells

Replace `lpflow_inputiso1p`/`lpflow_isobufsrc` instances with `a21o_1` or `mux2_1`. These
power-domain primitives are inappropriate in a synchronous compute path.

## Clock target

Keep 300 MHz — the path analysis confirms it is achievable with pipelining. Dropping to
150 MHz would waste half the chiplet's compute bandwidth.

## Timeline

RTL changes by May 21; re-synthesis by May 23 (before M3 due May 24).
