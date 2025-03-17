cmake_minimum_required(VERSION 3.16)
project(benchmark_f16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(KLEIDIAI_PATH ../../../kleidiai/)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH}
)

# Files requires to build the executable
add_executable(benchmark_f16
    benchmark_f16.cpp
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
)

# Files requires to build the executable
add_executable(benchmark_f16_scaling
    benchmark_f16_scaling.cpp#
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
)

add_executable(kai_f16_inf
    kai_f16_inf.cpp#
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
)
target_compile_options(benchmark_f16
    PRIVATE -march=armv8.4-a+fp16+sve -mtune=neoverse-v1
)

target_compile_options(benchmark_f16_scaling
    PRIVATE -march=armv8.4-a+fp16+sve -mtune=neoverse-v1
)

target_compile_options(kai_f16_inf
    PRIVATE -march=armv8.4-a+fp16+sve -mtune=neoverse-v1
)

target_compile_definitions(benchmark_f16
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
