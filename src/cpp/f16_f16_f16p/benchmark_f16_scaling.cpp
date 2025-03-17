#include <iostream>
#include <vector>
#include <fstream>
#include "../common/sizes.cpp"
#include "kernel.cpp"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>




int main() {
    // Open CSV file for writing
    std::ofstream csv_file("../../results/f16_scaling_results.csv");
    // Write header
    csv_file << "Size,Latency(us)\n";

    for (int size: sizes) {
        std::vector<float16_t> X(size * size);
        std::vector<float16_t> W(size * size);
        std::vector<float16_t> Y(size * size, 0.0f);  // Initialize with zeros

        std::generate(X.begin(), X.end(), []() { return static_cast<float16_t>(rand()) / RAND_MAX; });
        std::generate(W.begin(), W.end(), []() { return static_cast<float16_t>(rand()) / RAND_MAX; });

        size_t M = size; 
        size_t N = size;
        size_t K = size;

        float16_t* lhs = X.data();
        float16_t* rhs = W.data();

        const size_t nr = ukernel.get_nr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();

        // In a single row, we pack nr bias values followed by K rows of nr RHS values
        const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
        const size_t rhs_packed_cols = nr + K * nr;
        const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float16_t));

        float16_t* rhs_packed = new float16_t[rhs_packed_size];

        const size_t lhs_stride = K * sizeof(float16_t);
        const size_t rhs_stride = N * sizeof(float16_t);
        const size_t dst_stride_row = N * sizeof(float16_t);
        const size_t dst_stride_col = sizeof(float16_t);
        //float* bias = new float[N];
        float16_t* bias = new float16_t[N];
        std::fill_n(bias, N, 0.0f);
        kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
            1, N, K, nr, kr, sr,  // Packing arguments
            rhs_stride,           // RHS stride
            rhs,                  // RHS
            bias,                 // Bias
            NULL,                 // Scale
            rhs_packed,           // RHS packed
            0, NULL);

        float16_t* dst = Y.data();
        auto start = std::chrono::high_resolution_clock::now();

        ukernel.run_matmul(
            M, N, K,           // Dimensions
            lhs,               // LHS
            lhs_stride,        // LHS stride
            rhs_packed,        // RHS packed
            dst,               // DST
            dst_stride_row,    // DST stride (row)
            dst_stride_col,    // DST stride (col)
            -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
        );

        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << "Time taken: " << duration << " microseconds" << std::endl;
        csv_file << size << "," << duration << "\n";
    }
    
    return 0;
}