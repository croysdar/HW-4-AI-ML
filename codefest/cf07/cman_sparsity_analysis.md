# CMAN — Sparsity Breakeven Analysis

N = 512, sparsity s = fraction of zeros.

## 1. Expressions

**(a) Dense MVM compute**

Each of the N² multiply-accumulate operations costs 2 FLOPs:

Dense FLOPs = 2N²

At N = 512: 2 × 512² = **524,288 FLOPs ≈ 524K**

**(b) Dense memory bytes**

4 bytes per FP32 element, N² elements:

Dense memory = 4N² bytes

At N = 512: **1,048,576 bytes ≈ 1 MB**

**(c) Sparse compute**

Only N²(1−s) non-zero entries require a MAC:

Sparse FLOPs = 2N²(1−s)

**(d) Sparse memory bytes (CSR)**

CSR uses three arrays:

- Values: N²(1−s) × 4 bytes
- Column indices: N²(1−s) × 4 bytes
- Row pointers: (N+1) × 4 bytes

Sparse memory = 8N²(1−s) + 4(N+1) bytes

## 2. FLOPs Speedup and 2× Breakeven

Speedup = Dense FLOPs / Sparse FLOPs:

Speedup(s) = 2N² / 2N²(1−s) = **1 / (1−s)**

For 2× speedup:

1/(1−s) = 2 → 1−s = 0.5 → **s = 0.50 (50%)**

## 3. Memory Breakeven Sparsity

Set dense bytes equal to sparse bytes. Dividing through by 4:

N² = 2N²(1−s) + (N+1)

Solving for s:

N² − N − 1 = 2N²(1−s)

s = 1 − (N²−N−1)/(2N²) = **(N²+N+1) / (2N²)**

At N = 512:

s = (262144 + 512 + 1) / 524288 = 262657 / 524288 ≈ **0.501**

Above s ≈ 0.501, CSR uses less memory than dense storage.

## 4. End-to-End Speedup at s = 0.9

For a memory-bandwidth-limited system, execution time scales with bytes loaded. Speedup = Dense bytes / Sparse bytes:

Speedup = 4N² / [8N²(1−s) + 4(N+1)] = N² / [2N²(1−s) + (N+1)]

At N = 512, s = 0.9:

Sparse bytes fraction = [2 × 262144 × 0.1 + 513] / 262144 = 52941.8 / 262144 ≈ **0.201**

Speedup = 1 / 0.201 ≈ **4.97× ≈ 5×**

As N → ∞ the row-pointer term vanishes and the speedup limit is 1 / (2(1−s)) = 1/(2 × 0.1) = **5×**, confirming the result.
