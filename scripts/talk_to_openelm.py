import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = "models/hf_models/OpenELM-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "apple/OpenELM-270M",
    trust_remote_code=True,
    torch_dtype=torch.float16  # <- use float32 on Pi
)

# Prompt
if not sys.stdin.isatty():
    prompt = sys.stdin.read().strip()
else:
    prompt = input("Enter your question: ").strip()

# Tokenize & move to CPU
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}
model.to("cpu")

# Stream response token-by-token
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.generate(**inputs, max_new_tokens=50, streamer=streamer)
