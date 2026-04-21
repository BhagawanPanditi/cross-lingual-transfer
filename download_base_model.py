import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

base_model_name = "meta-llama/Llama-3.2-3B"
output_path = "./llama3b"

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN" 

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN 
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    token=HF_TOKEN
)

print(f"Base vocab size: {model.config.vocab_size}")
print(f"embed_tokens shape: {model.model.embed_tokens.weight.shape}")
print(f"lm_head shape: {model.lm_head.weight.shape}")

print(f"\nSaving base model to {output_path}...")
os.makedirs(output_path, exist_ok=True)

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("Done!")
