# GEMM Analysis: Naive vs. Tiled (T=8) on NVIDIA T4

## (a) Why the Naive Kernel is Memory-Bound

In my tests on the NVIDIA T4, the naive kernel was heavily memory-bound, which is expected given the access pattern. Since each thread calculates one output element by looping through N=1024 values, it ends up re-fetching elements from Matrix B over and over. This resulted in roughly 8 GB of DRAM traffic and a very low arithmetic intensity of 0.25 FLOPs/byte. On a T4, the "ridge point" where a kernel becomes compute-bound is approximately 25 FLOPs/byte. Because 0.25 is so far to the left of that ridge, the GPU spends most of its time waiting for data rather than calculating, which explains why I only achieved 1.51 GFLOP/s despite the T4’s 8.1 TFLOPS theoretical peak.

## (b) How Tiling Reduces DRAM Traffic

By implementing tiling with T=8, I used **shared** memory to create a local buffer for the threads. Mathematically, this should be a help a lot; by loading 8x8 tiles and having the threads cooperatively reuse that data, the DRAM traffic is reduced by a factor of T (8), bringing it down to about 1 GB. This improved the arithmetic intensity to 2.0 FLOPs/byte. However, the performance actually stayed flat (or even dipped slightly) at 1.50 GFLOP/s.

## (c) Did Tiling Achieve the Expected Improvement?

No. Looking at the Nsight Compute data, the issue is clearly low hardware utilization. A tile size of T=8 only puts 64 threads in each work block. The T4 hardware needs a much higher degree of parallelism to hide the time it takes to fetch data from memory; with only 64 threads, the GPU’s processing cores simply can’t stay busy. Nsight showed SM throughput dropping from 62.5% to 55.2%, proving that the overhead of managing shared memory actually slowed things down because there wasn't enough work happening at once to justify it. Even though the kernel is "closer to the ridge" on the roofline, it remains stalled because there aren't enough active threads to keep the engine running while waiting on memory. To actually see a performance boost, I would likely need to increase the tile size to T=16 or T=32 to better saturate the GPU.
