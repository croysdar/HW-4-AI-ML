1.

| Layer | Number of Nodes | MAC operations |
| ----- | --------------- | -------------- |
| 1     | 784             | 0              |
| 2     | 256             | 200704         |
| 3     | 128             | 32768          |
| 4     | 10              | 1280           |

2.

Total MAC operations:
234752

3.

Total trainable parameters
234752

4.

Total weight memory
4 \* 234752 = 939008 bytes

5.

Total activation memory
784 + 256 + 128 + 10 = 1178
1178 \* 4 = 4712 bytes

6.

Arithmetic intensity
2 \* (234752) / (939008 + 4712) = 0.4975 FLOP/byte
