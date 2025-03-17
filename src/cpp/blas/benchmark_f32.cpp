#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cblas.h>
#include <algorithm>
#include "../common/sizes.cpp"

int main() {
    // Open CSV file for writing
    std::ofstream csv_file("../../results/blas_f32_scaling_results.csv");
    // Write header
    csv_file << "Size,Latency(us)\n";


    for (int size : sizes) {
        // Allocate memory for matrices X, W, and Y
        std::vector<float> X(size * size);
        std::vector<float> W(size * size);
        std::vector<float> Y(size * size, 0.0f);  // Initialize with zeros

        // Populate X and W with random values
        std::generate(X.begin(), X.end(), []() { return static_cast<float>(rand()) / RAND_MAX; });
        std::generate(W.begin(), W.end(), []() { return static_cast<float>(rand()) / RAND_MAX; });

        const int M = size;
        const int N = size;
        const int K = size;

        const float alpha = 1.0f; // Scaling factor for A*B
        const float beta = 0.0f;  // Scaling factor for C

        // Warmup: Run a single matmul to ensure everything is loaded into memory
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    X.data(), K,
                    W.data(), N,
                    beta,
                    Y.data(), N);

        // Measure time for matrix multiplication
        float* X_ptr = X.data();
        float* W_ptr = W.data(); 
        float* Y_ptr = Y.data();
        auto start = std::chrono::high_resolution_clock::now();

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    X_ptr, K,
                    W_ptr, N,
                    beta,
                    Y_ptr, N);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Print and log the results
        //std::cout << "Size: " << size << ", Time taken: " << duration << " milliseconds" << std::endl;
        csv_file << size << "," << duration << "\n";
    }

    csv_file.close();
    return 0;
}
