#! /bin/bash

llama-bench -m models/gguf_models/OpenELM-3B-Instruct-f32.gguf -p 128 -n 1 --threads 1