#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1024
#define T 8

__global__ void gemm_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[T][T];
    __shared__ float sB[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (n + T - 1) / T;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * T + threadIdx.x;
        int b_row = t * T + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < T; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (N + T - 1) / T);

    // warm-up
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * (double)N * N * N;
    double gflops = flops / (ms * 1e6);
    printf("Tiled GEMM  N=%d  T=%d  time=%.3f ms  Achieved=%.2f GFLOP/s\n", N, T, ms, gflops);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // CPU verification on a 4x4 corner
    printf("Verifying top-left 4x4 corner against CPU reference...\n");
    float ref[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++)
                s += h_A[i * N + k] * h_B[k * N + j];
            ref[i][j] = s;
        }
    float max_err = 0.0f;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float err = fabsf(h_C[i * N + j] - ref[i][j]);
            if (err > max_err) max_err = err;
        }
    printf("Max absolute error (4x4 corner): %e\n", max_err);
    printf("Verification %s\n", max_err < 1e-2f ? "PASSED" : "FAILED");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
