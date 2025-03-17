# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(BlasScalingBenchmark VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the executables
add_executable(blas_scaling_benchmark benchmark_f32.cpp)
add_executable(blas_f32_inf f32_inf.cpp)

# Find and link BLAS library
find_package(BLAS REQUIRED)

if (BLAS_FOUND)
    message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
    target_link_libraries(blas_scaling_benchmark PUBLIC ${BLAS_LIBRARIES})
    target_link_libraries(blas_f32_inf PUBLIC ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS library not found!")
endif()

# Include BLAS directories if required
if (BLAS_INCLUDE_DIRS)
    message(STATUS "Using BLAS include directories: ${BLAS_INCLUDE_DIRS}")
    target_include_directories(blas_scaling_benchmark PUBLIC ${BLAS_INCLUDE_DIRS})
    target_include_directories(blas_f32_inf PUBLIC ${BLAS_INCLUDE_DIRS})
endif()

# Add architecture-specific compiler flags (NEON or SVE)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") # Check if ARM architecture
    message(STATUS "ARM architecture detected; adding NEON/SVE flags")
    target_compile_options(blas_scaling_benchmark PRIVATE -march=armv8.4-a+sve -mtune=neoverse-v1)
    target_compile_options(blas_f32_inf PRIVATE -march=armv8.4-a+sve -mtune=neoverse-v1)
else()
    message(WARNING "Non-ARM architecture detected; skipping NEON/SVE flags")
endif()

# Optional: Specify installation rules (if needed)
install(TARGETS blas_scaling_benchmark blas_f32_inf DESTINATION bin)
