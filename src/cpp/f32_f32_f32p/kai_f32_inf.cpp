#include <iostream>
#include <vector>
#include "kernel.cpp"
#include "kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>


void loadMatrix(const char* filename, float* matrix, size_t rows, size_t cols) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return;
    }

    size_t elements_read = fread(matrix, sizeof(float), rows * cols, file);
    if (elements_read != rows * cols) {
        fprintf(stderr, "Error: Only %zu elements could be read.\n", elements_read);
    }
    fclose(file);
}



int main() {
    // Declare matrix dimensions
    const size_t activation_rows = 6, activation_cols = 1280;
    const size_t weight_rows = 1280, weight_cols = 32000;

    std::vector<float> X(activation_rows * activation_cols);
    std::vector<float> W(weight_rows * weight_cols);
    std::vector<float> Y(activation_rows * weight_cols, 0.0f);  // Initialize with zeros

    size_t M = activation_rows; 
    size_t N = weight_cols;
    size_t K = activation_cols;

    loadMatrix("../../assets/x_fp32.bin", X.data(), activation_rows, activation_cols);
    loadMatrix("../../assets/w_fp32.bin", W.data(), weight_rows, weight_cols);

    float* lhs = X.data();
    float* rhs = W.data();

    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    // In a single row, we pack nr bias values followed by K rows of nr RHS values
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
    const size_t rhs_packed_cols = nr + K * nr;
    const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float));

    float* rhs_packed = new float[rhs_packed_size];

    const size_t lhs_stride = K * sizeof(float);
    const size_t rhs_stride = N * sizeof(float);
    const size_t dst_stride_row = N * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    //float* bias = new float[N];
    float* bias = new float[N];
    std::fill_n(bias, N, 0.0f);
    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
        1, N, K, nr, kr, sr,  // Packing arguments
        rhs_stride,           // RHS stride
        rhs,                  // RHS
        bias,                 // Bias
        NULL,                 // Scale
        rhs_packed,           // RHS packed
        0, NULL);

    float* dst = Y.data();
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
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
    }

    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
    
    return 0;
}