#include <iostream>
#include <vector>
#include "kernel.cpp"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>


void loadMatrix(const char* filename, float16_t* matrix, size_t rows, size_t cols) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return;
    }

    // Temporary buffer for float32 values
    std::vector<float> temp_buffer(rows * cols);
    
    size_t elements_read = fread(temp_buffer.data(), sizeof(float), rows * cols, file);
    if (elements_read != rows * cols) {
        fprintf(stderr, "Error: Only %zu elements could be read.\n", elements_read);
    }
    fclose(file);

    // Convert float32 to float16
    for (size_t i = 0; i < rows * cols; i++) {
        matrix[i] = float16_t(temp_buffer[i]);
    }
}



int main() {
    // Declare matrix dimensions
    const size_t activation_rows = 6, activation_cols = 1280;
    const size_t weight_rows = 1280, weight_cols = 32000;

    std::vector<float16_t> X(activation_rows * activation_cols);
    std::vector<float16_t> W(weight_rows * weight_cols);
    std::vector<float16_t> Y(activation_rows * weight_cols, 0.0f);  // Initialize with zeros

    size_t M = activation_rows; 
    size_t N = weight_cols;
    size_t K = activation_cols;

    loadMatrix("../../assets/x_fp32.bin", X.data(), activation_rows, activation_cols);
    loadMatrix("../../assets/w_fp32.bin", W.data(), weight_rows, weight_cols);

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
    float16_t* dst = Y.data();
    
    
    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
        1, N, K, nr, kr, sr,  // Packing arguments
        rhs_stride,           // RHS stride
        rhs,                  // RHS
        bias,                 // Bias
        NULL,                 // Scale
        rhs_packed,           // RHS packed
        0, NULL);

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
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
    
    return 0;
}