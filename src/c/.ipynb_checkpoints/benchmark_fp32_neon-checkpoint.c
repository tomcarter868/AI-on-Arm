#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <malloc.h>
#include "kernels/fp32_neon.c"
#include "sizes.c"

// Function to get time in seconds with high resolution
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

int main() {
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    FILE* fp;

    fp = fopen("results/fp32_neon_latency_results.csv", "w");
    if (fp == NULL) {
        printf("Error opening file for writing\n");
        return 1;
    }
    fprintf(fp, "Matrix Size,Latency (seconds)\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        float* A;
        float* B;
        float* C;

        posix_memalign((void**)&A, 16, N * N * sizeof(float));
        posix_memalign((void**)&B, 16, N * N * sizeof(float));
        posix_memalign((void**)&C, 16, N * N * sizeof(float));

        // Initialize matrices with some values
        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0f;
            B[i] = 1.0f;
        }

        // Warm-up iterations
        for (int warmup = 0; warmup < 3; warmup++) {
            matmul_fp32_neon(A, B, C, N);
        }

        double start = get_time();
        matmul_fp32_neon(A, B, C, N);
        double end = get_time();

        double time_taken = end - start;
        printf("FP32 NEON Matrix Multiplication (Size %d): %f seconds\n", N, time_taken);
        fprintf(fp, "%d,%f\n", N, time_taken);

        free(A);
        free(B);
        free(C);
    }
    fclose(fp);
    return 0; 
}