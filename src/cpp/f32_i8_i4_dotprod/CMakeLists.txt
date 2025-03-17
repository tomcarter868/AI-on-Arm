cmake_minimum_required(VERSION 3.16)
project(benchmark_dotprod)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(KLEIDIAI_PATH ../../../kleidiai/)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH}
)



add_executable(benchmark_dotprod
    benchmark_i8_dotprod.cpp
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
    ${MATMUL_PACK_PATH}/kai_lhs_quant_pack_qsi8d32p_f32.c
)

add_executable(benchmark_dotprod_scaling
    benchmark_i8_dotprod_scaling.cpp
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
    ${MATMUL_PACK_PATH}/kai_lhs_quant_pack_qsi8d32p_f32.c
)

add_executable(kai_i8_dotprod_inf
    kai_i8_dotprod_inf.cpp
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
    ${MATMUL_PACK_PATH}/kai_lhs_quant_pack_qsi8d32p_f32.c
)


target_compile_options(benchmark_dotprod
    PRIVATE -march=armv8.4-a+sve+dotprod -mtune=neoverse-v1
)

target_compile_options(benchmark_dotprod_scaling
    PRIVATE -march=armv8.4-a+sve+dotprod -mtune=neoverse-v1
)

target_compile_options(kai_i8_dotprod_inf
    PRIVATE -march=armv8.4-a+sve+dotprod -mtune=neoverse-v1
)

target_compile_definitions(benchmark_dotprod
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
