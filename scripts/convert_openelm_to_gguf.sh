#!/bin/bash

python llama.cpp/convert_hf_to_gguf.py models/hf_models/OpenELM-3B-Instruct/ --outfile models/gguf_models/OpenELM-3B-Instruct-{ftype}.gguf --outtype f32

llama-quantize models/gguf_models/OpenELM-3B-Instruct-f32.gguf models/gguf_models/OpenELM-3B-Instruct-f16.gguf F16
llama-quantize models/gguf_models/OpenELM-3B-Instruct-f32.gguf models/gguf_models/OpenELM-3B-Instruct-q8_0.gguf Q8_0 
llama-quantize models/gguf_models/OpenELM-3B-Instruct-f32.gguf models/gguf_models/OpenELM-3B-Instruct-q4_0.gguf Q4_0 

