## Top 5 Layers by Mult-Adds

| Name           | MACs        | Parameter Count |
| -------------- | ----------- | --------------- |
| conv1          | 118,013,952 | 9,408           |
| layer4.1.conv2 | 115,605,504 | 2,359,296       |
| layer4.1.conv1 | 115,605,504 | 2,359,296       |
| layer4.0.conv2 | 115,605,504 | 2,359,296       |
| layer3.1.conv2 | 115,605,504 | 589,824         |

There are 14 layers with the same MAC count of 115,605,504. To choose which to show as the top 5, I looked at params count.

## Arithmetic Intensity of most MAC-intensive layer

### DRAM traffic

| Tensor             | Shape              |         Bytes |
| ------------------ | ------------------ | ------------: |
| Input activations  | 1 × 3 × 224 × 224  |       602,112 |
| Weights            | 64 × 3 × 7 × 7     |        37,632 |
| Output activations | 1 × 64 × 112 × 112 |     3,211,264 |
| **Total**          |                    | **3,851,008** |

### Result

```
Arithmetic Intensity = 2 * MACs / bytes
                     = 2 * 118,013,952 / 3,851,008
                     = 61.3 FLOPs/byte
```
