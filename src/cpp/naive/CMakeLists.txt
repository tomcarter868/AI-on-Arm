cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NaiveBenchmark)

# Specify C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
add_executable(benchmark_naive benchmark_naive.cpp kernel.cpp)
