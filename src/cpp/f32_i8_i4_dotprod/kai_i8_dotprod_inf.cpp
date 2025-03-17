#include <iostream>
#include <vector>
#include "kernel.cpp"
#include "kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

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



static inline size_t num_blocks_per_row(size_t k, size_t bl) {
    return k / bl;
}

static inline size_t num_bytes_per_block_qs8c32(size_t bl) {
    return bl + sizeof(int16_t);
}

static inline size_t num_bytes_per_block_qs4c32(size_t bl) {
    return (bl / 2) + sizeof(int16_t);
}


static void quant_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs4c32) {
    const size_t num_blocks_row = num_blocks_per_row(k, bl);
    const size_t num_bytes_block = num_bytes_per_block_qs4c32(bl);
    const size_t dst_stride = num_blocks_row * num_bytes_block;

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        uint8_t* dst_ptr = (uint8_t*)rhs_qs4c32 + row_idx * dst_stride;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;
            float max = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const float src0_0 = src_ptr[block_idx * bl + b];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                    max = src0_0;
                }
            }

            const float scale = max / -8.0;
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            // Store the scale at the beginning of the block
            *((uint16_t*)dst_ptr) = kai_cast_f16_f32(scale);
            dst_ptr += sizeof(uint16_t);

            const size_t block_size = 32;
            const size_t num_subblocks = bl / 32;

            for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
                for (size_t i = 0; i < block_size / 2; ++i) {
                    const size_t src_base_addr = block_idx * bl + i + subblock_idx * block_size;
                    float v0_f32 = src_ptr[src_base_addr];
                    float v1_f32 = src_ptr[src_base_addr + block_size / 2];

                    v0_f32 *= recip_scale;
                    v1_f32 *= recip_scale;

                    const uint8_t v0_u8 = (uint8_t)std::min((int8_t)15, (int8_t)(v0_f32 + 8.5f));
                    const uint8_t v1_u8 = (uint8_t)std::min((int8_t)15, (int8_t)(v1_f32 + 8.5f));

                    const uint8_t rhs_v0 = (v1_u8 << 4) | v0_u8;

                    dst_ptr[0] = rhs_v0;
                    dst_ptr += sizeof(uint8_t);
                }
            }
        }
    }
};

static void ref_quant_qs8d32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs8c32) {
    const size_t num_blocks_row = num_blocks_per_row(k, bl);
    const size_t num_bytes_block = num_bytes_per_block_qs8c32(bl);
    const size_t dst_stride = num_blocks_row * num_bytes_block;

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        int8_t* dst_ptr = (int8_t*)rhs_qs8c32 + row_idx * dst_stride;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const float src0_0 = src_ptr[block_idx * bl + b];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                }
            }

            const float scale = amax / ((1 << 7) - 1);
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            // Store the scale at the beginning of the block
            *((uint16_t*)dst_ptr) = kai_cast_f16_f32(scale);
            dst_ptr += sizeof(uint16_t);

            const size_t block_size = 32;
            const size_t num_subblocks = bl / 32;

            for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
                for (size_t i = 0; i < block_size; ++i) {
                    const size_t src_base_addr = block_idx * bl + i + subblock_idx * block_size;
                    float v0_f32 = src_ptr[src_base_addr];

                    v0_f32 *= recip_scale;

                    dst_ptr[0] = roundf(v0_f32);
                    dst_ptr += sizeof(int8_t);
                }
            }
        }
    }
};


static void ref_matmul_f32_qs8d32_qs4c32(
    size_t m, size_t n, size_t k, size_t bl, const int8_t* lhs_qa8d32, const uint8_t* rhs_qs4c32, float* dst_f32,
    float scalar_min, float scalar_max) {
    const size_t num_blocks_row = num_blocks_per_row(k, bl);
    const size_t num_bytes_block_qs4c32 = num_bytes_per_block_qs4c32(bl);
    const size_t num_bytes_block_qs8c32 = num_bytes_per_block_qs8c32(bl);

    const size_t lhs_stride = num_blocks_row * num_bytes_block_qs8c32;
    const size_t rhs_stride = num_blocks_row * num_bytes_block_qs4c32;

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        const int8_t* lhs_ptr_start = lhs_qa8d32 + row_idx * lhs_stride;
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            // Main f32 accumulator
            float main_acc = 0.0f;

            const size_t block_size = 32;
            const size_t num_subblocks = bl / 32;

            for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
                const int8_t* lhs_ptr = lhs_ptr_start;
                const uint8_t* rhs_ptr = rhs_qs4c32 + col_idx * rhs_stride;

                lhs_ptr += block_idx * num_bytes_block_qs8c32;
                rhs_ptr += block_idx * num_bytes_block_qs4c32;

                for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
                    int32_t temp_acc = 0;

                    // Get the LHS/RHS quantization scale stored at the
                    // beginning of each block
                    const float lhs_scale = kai_cast_f32_f16(*(const uint16_t*)lhs_ptr);
                    const float rhs_scale = kai_cast_f32_f16(*(const uint16_t*)rhs_ptr);

                    lhs_ptr += sizeof(uint16_t);
                    rhs_ptr += sizeof(uint16_t);

                    for (size_t i = 0; i < block_size / 2; ++i) {
                        // Get the LHS values
                        const int32_t lhs_v0 = (int32_t)lhs_ptr[0];
                        const int32_t lhs_v1 = (int32_t)lhs_ptr[block_size / 2];

                        // Get the RHS values
                        const uint8_t rhs_byte = rhs_ptr[0];

                        // Unpack the RHS values
                        const int32_t rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
                        const int32_t rhs_v1 = (((int32_t)(rhs_byte >> 4)) - 8);

                        temp_acc += lhs_v0 * rhs_v0;
                        temp_acc += lhs_v1 * rhs_v1;

                        lhs_ptr += 1;
                        rhs_ptr += 1;
                    }

                    main_acc += temp_acc * lhs_scale * rhs_scale;
                }
            }

            main_acc = std::max(main_acc, scalar_min);
            main_acc = std::min(main_acc, scalar_max);

            dst_f32[0] = main_acc;
            dst_f32 += 1;
        }
    }
};




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

    const size_t mr = ukernel.get_mr();
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    const size_t bl = 32;  // Block length. It must be 32
    const size_t m = activation_rows;
    const size_t n = weight_cols;
    const size_t k = activation_cols;
    const size_t seed_lhs = 4568;
    const size_t seed_rhs = seed_lhs + 4;
    
    const size_t num_blocks = k / bl;
    const size_t num_bytes_per_block_qs4c32 = (bl / 2) + sizeof(int16_t);
    const size_t num_bytes_per_block_qs8c32 = bl + sizeof(int16_t);

    const size_t rhs_native_size_qs4c32 = n * num_blocks * num_bytes_per_block_qs4c32;
    uint8_t* rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];

    quant_qs4c32_f32(n, k, bl, (const float*)W.data(), (uint8_t*)rhs_native_mtx_qs4c32);


    const size_t lhs_ref_size_qa8d32 = m * num_blocks * num_bytes_per_block_qs8c32;
    const size_t dst_ref_size_f32 = m * n * sizeof(float);

    uint8_t* lhs_ref_mtx_qa8d32 = new uint8_t[lhs_ref_size_qa8d32];
    uint8_t* dst_ref_mtx_f32 = new uint8_t[dst_ref_size_f32];

    ref_quant_qs8d32_f32(m, k, bl, (const float*)X.data(), (uint8_t*)lhs_ref_mtx_qa8d32);




    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, bl);
    const size_t dst_size = ukernel.get_dst_size(m, n);

    uint8_t* lhs_packed_mtx_qs8d32 = new uint8_t[lhs_packed_size];
    uint8_t* rhs_packed_mtx_qs4c32 = new uint8_t[rhs_packed_size];
    uint8_t* dst_act_mtx_f32 = new uint8_t[dst_size];

    struct kai_rhs_pack_qs4cxs1s0_param params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    /*
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
            1, n, k,                                  // Dimensions
            nr, kr, sr,                               // Packing arguments
            bl,                                       // Block length
            (const uint8_t*)(rhs_native_mtx_qs4c32),  // RHS
            NULL,                                     // Bias
            rhs_packed_mtx_qs4c32,                    // RHS packed
            0, &params);
    */
   ref_matmul_f32_qs8d32_qs4c32(
        m, n, k, bl, (const int8_t*)lhs_ref_mtx_qa8d32, (const uint8_t*)rhs_native_mtx_qs4c32, (float*)dst_ref_mtx_f32,
        -FLT_MAX, FLT_MAX);

    // If the RHS matrix contains constant values, the packing can be performed
    // only once

    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset = ukernel.get_lhs_packed_offset(0, k, bl);
    const size_t rhs_offset = ukernel.get_rhs_packed_offset(0, k, bl);
    const size_t dst_offset = ukernel.get_dst_offset(0, 0, dst_stride);

    const void* lhs_ptr = (const void*)((const char*)lhs_packed_mtx_qs8d32 + lhs_offset);
    const void* rhs_ptr = (const void*)((const char*)rhs_packed_mtx_qs4c32 + rhs_offset);
    float* dst_ptr = (float*)((uint8_t*)dst_act_mtx_f32 + dst_offset);
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        ukernel.run_matmul(
            m, n, k, bl,       // Dimensions
            lhs_ptr,           // LHS packed
            rhs_ptr,           // RHS packed
            dst_ptr,           // DST
            dst_stride,        // DST stride (row)
            sizeof(float),     // DST stride (col)
            -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
    
    return 0;
}