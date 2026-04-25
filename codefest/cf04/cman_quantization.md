# CMAN — Manual INT8 Symmetric Quantization

## 1. Scale Factor

- max|W| = **2.31** (element W[2,3])
- S = 2.31 / 127 = **0.018189**

## 2. Quantize — W_q (INT8)

W_q = round(W / S), clamped to [−128, 127]

|     |     |      |      |
| --- | --- | ---- | ---- |
| 47  | −66 | 19   | 115  |
| −4  | 50  | −103 | 7    |
| 85  | 2   | −24  | −127 |
| −10 | 57  | 42   | 30   |
|     |     |      |      |

## 3. Dequantize — W_deq (FP32)

W_deq = W_q × S

|         |         |         |         |
| ------- | ------- | ------- | ------- |
| 0.8549  | −1.2005 | 0.3456  | 2.0917  |
| −0.0728 | 0.9094  | −1.8735 | 0.1273  |
| 1.5461  | 0.0364  | −0.4365 | −2.3100 |
| −0.1819 | 1.0368  | 0.7639  | 0.5457  |
|         |         |         |         |

## 4. Error Analysis

Per-element absolute error |W − W_deq|:

|          |          |          |          |
| -------- | -------- | -------- | -------- |
| 0.004882 | 0.000472 | 0.005591 | 0.008268 |
| 0.002756 | 0.000551 | 0.006535 | 0.007323 |
| 0.003937 | 0.006378 | 0.003465 | 0.000000 |
| 0.001890 | 0.006772 | 0.006063 | 0.004331 |
|          |          |          |          |

- **Largest error:** W[0,3] = 2.10 → error = **0.008268**
- **MAE** = (sum of all 16 errors) / 16 = 0.069214 / 16 = **0.004326**

## 5. Bad Scale Experiment (S_bad = 0.01)

W_q with S_bad (clamped to [−128, 127]):

|     |      |      |      |
| --- | ---- | ---- | ---- |
| 85  | −120 | 34   | 127  |
| −7  | 91   | −128 | 12   |
| 127 | 3    | −44  | −128 |
| −18 | 103  | 77   | 55   |
|     |      |      |      |

Per-element absolute error |W − W_deq_bad|:

|      |      |      |      |
| ---- | ---- | ---- | ---- |
| 0.00 | 0.00 | 0.00 | 0.83 |
| 0.00 | 0.00 | 0.60 | 0.00 |
| 0.28 | 0.00 | 0.00 | 1.03 |
| 0.00 | 0.00 | 0.00 | 0.00 |
|      |      |      |      |

- **MAE** (S_bad) = 2.74 / 16 = **0.171**

**One-sentence analysis:**

When S is too small, large-magnitude values exceed the INT8 range and the clamping to ±127/128 cuts them off, causing large errors in the dequantized output.
