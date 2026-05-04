PE Array

[5][6]
[7][8]

| Cycle | Input to row 0 | Input to row 1 | PE[0][0] | PE[0][1] | PE[1][0] | PE[1][1] | Output values |
| ----- | -------------- | -------------- | -------- | -------- | -------- | -------- | ------------- |
| 1     | 1              |                | 1x5      |          |          |          |               |
| 2     | 3              | 2              | 3x5      | 1x6      | 2x7 + 5  |          |               |
| 3     |                | 4              |          | 3x6      | 4x7 + 15 | 2x8 + 6  | 19 [ ]        |
| 4     |                |                |          |          |          | 4x8 + 18 | [43] [22]     |
| 5     |                |                |          |          |          |          | [ ] [50]      |

**Note on timing:** The cycle table shows 5 cycles. Although all MACs complete by the end of cycle 4, the final result (C[1][1]=50) is not available to be latched and read out until cycle 5. In a real clocked pipeline, a value computed during cycle N is captured in an output register at the rising edge of cycle N+1. So the realistic latency from first input to last output is 5 cycles, not 4.

3.  a. **Total MAC operations**

2 MAC ops per output value, as seen in the table, so 8.

3. b. **Input reuse amount**

Each input value was reused two times

3. c. **Number of off chip-accesses**

For A and B: Each element is accessed once, so 4, and 4 is 8

For C: Each element is accessed once, so 4

Total: 12

4. **If this were output stationary instead, which values would stay fixed?**

The partial sums would remain fixed in place rather than the weights. The weights would then stream through the array as well.
