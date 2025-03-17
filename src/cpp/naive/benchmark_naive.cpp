#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "../common/sizes.cpp"
#include "kernel.h"

void loadMatrix(const std::string& filename, float* matrix, size_t rows, size_t cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    file.read(reinterpret_cast<char*>(matrix), rows * cols * sizeof(float));
    if (file.gcount() != rows * cols * sizeof(float)) {
        std::cerr << "Error: Only " << file.gcount() << " bytes could be read." << std::endl;
    }
    file.close();
}

int main() {
    // Declare matrix dimensions
    const size_t activation_rows = 23, activation_cols = 3072;
    const size_t weight_rows = 3072, weight_cols = 32000;

    std::vector<float> X(activation_rows * activation_cols);
    std::vector<float> W(weight_rows * weight_cols);
    std::vector<float> Y(activation_rows * weight_cols, 0.0f);  // Initialize with zeros

    loadMatrix("../../assets/x_fp32.bin", X.data(), activation_rows, activation_cols);
    loadMatrix("../../assets/w_fp32.bin", W.data(), weight_rows, weight_cols);


    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply_naive(X.data(), W.data(), Y.data(), activation_rows, activation_cols, weight_cols);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "Time taken: " << duration << " seconds" << std::endl;

    return 0;
}
