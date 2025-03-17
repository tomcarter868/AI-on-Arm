cmake_minimum_required(VERSION 3.16)
project(benchmark_f32)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KLEIDIAI_PATH ../../../kleidiai/)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/)

# Enable Assembly language
enable_language(ASM)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH}
)

# Assembly file path
set(ASM_FILE ${MATMUL_PATH}/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_asm.S)

# Files required to build the executables
add_executable(benchmark_f32
    benchmark_f32.cpp
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    ${ASM_FILE}  # Add Assembly file
)

add_executable(benchmark_f32_scaling
    benchmark_f32_scaling.cpp
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    ${ASM_FILE}  # Add Assembly file
)

add_executable(kai_f32_inf
    kai_f32_inf.cpp
    kernel.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    ${ASM_FILE}  # Add Assembly file
)

# Compiler flags for ARMv8.4-A and Neoverse V1
set(ARM_FLAGS "-march=armv8.4-a" "-mtune=neoverse-v1")

target_compile_options(benchmark_f32 PRIVATE ${ARM_FLAGS})
target_compile_options(benchmark_f32_scaling PRIVATE ${ARM_FLAGS})
target_compile_options(kai_f32_inf PRIVATE ${ARM_FLAGS})

# Set correct properties for the assembly file
set_source_files_properties(${ASM_FILE} PROPERTIES LANGUAGE ASM)

target_compile_definitions(benchmark_f32
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
