# DRAM Traffic Analysis

ECE 510 Spring 2026 — Codefest 03

---

## 1. Naive Triple Loop (ijk order)

Accesses per element of B: Each element of matrix B is accessed N times (once for each row of matrix A it is multiplied with).

Since each element in both N×N matrices is accessed N times, the total element accesses across A and B are:

```
Total accesses = 2 × N³ = 2 × 32³ = 65,536 accesses
```

Total DRAM traffic:

```
Traffic = (2 × N³) × bytes per FP32
        = 65,536 × 4
        = 262,144 bytes  (256 KB)
```

---

## 2. Tiled Loop (Tile size T = 8)

Assuming perfect reuse — each element is loaded into fast memory exactly once — the total traffic is simply the size of matrices A and B:

```
Traffic = (2 × N²) × bytes per FP32
        = 2 × 1,024 × 4
        = 8,192 bytes  (8 KB)
```

---

## 3. Traffic Ratio

```
Ratio = Naive Traffic / Tiled Traffic
      = 262,144 / 8,192
      = 32  (= N)
```

In the tiled approach, each element of A and B is loaded from DRAM into fast memory exactly once. In the naive approach, each element is reloaded N times — once per outer-loop iteration — producing an N× traffic penalty.

---

## 4. Execution Time and Bottlenecks

**Total Compute (FLOPs):**

```
FLOPs = 2 × N³ = 2 × 32³ = 65,536 FLOP
```

**Compute Time:**

```
T_compute = 65,536 FLOP / (10 × 10¹² FLOP/s)
           = 6.55 ns
```

**Naive DRAM Access Time:**

```
T_mem_naive = 262,144 bytes / (320 × 10⁹ bytes/s)
            = 819.2 ns
```

**Tiled DRAM Access Time:**

```
T_mem_tiled = 8,192 bytes / (320 × 10⁹ bytes/s)
            = 25.6 ns
```

**Bound Classifications:**

| Case  | Compute Time | Memory Time | Bottleneck   |
| ----- | ------------ | ----------- | ------------ |
| Naive | 6.55 ns      | 819.2 ns    | Memory-bound |
| Tiled | 6.55 ns      | 25.6 ns     | Memory-bound |

- **Naive case:** Strictly memory-bound — memory time is ~125× larger than compute time.
- **Tiled case:** Still memory-bound, but significantly closer to the ridge of the roofline compared to the naive case. Tiling reduces memory traffic by N = 32×, closing most of the gap between memory and compute time.
