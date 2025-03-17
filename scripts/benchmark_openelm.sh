#!/bin/bash

# benchmark the floating point 16 model and write it into 
mkdir openelm_results
llama.cpp/build/bin/llama-bench -m models/gguf_models/OpenELM-3B-Instruct-q4_0.gguf -p 12 -n 6 --threads 1,2,3,4 | tee openelm_results/q4_pi_results.txt
llama.cpp/build/bin/llama-bench -m models/gguf_models/OpenELM-3B-Instruct-q8_0.gguf -p 12 -n 6 --threads 1,2,3,4 | tee openelm_results/q8_pi_results.txt 

# NOTE: we are not benchmarking f16 on Raspberry Pi5 as the benchmark takes too long
# llama.cpp/build/bin/llama-bench -m models/gguf_models/OpenELM-3B-Instruct-f16.gguf -p 12 -n 6 --threads 1,2,3,4 | tee openelm_results/f16_pi_results.txt 

 
