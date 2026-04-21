# prepare_login_node.py
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

model_name = "./llama3b"
save_dir = "./wiki_language_tokens"
os.makedirs(save_dir, exist_ok=True)

print("Downloading model & tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_wiki(tokenizer, language="as"):
    dataset_path = f"20231101.{language}"
    dataset = load_dataset("wikimedia/wikipedia", dataset_path, split="train")

    max_tokens = 100_000
    batch_size = 1000
    token_ids = []
    running = 0

    for idx in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[idx:idx+batch_size]["text"]
        tok = tokenizer(batch, add_special_tokens=False)

        for ids in tok["input_ids"]:
            if running + len(ids) > max_tokens:
                remain = max_tokens - running
                token_ids.extend(ids[:remain])
                running = max_tokens
                break
            else:
                token_ids.extend(ids)
                running += len(ids)

        if running >= max_tokens:
            break

    token_ids = torch.tensor(token_ids, dtype=torch.int32)
    fname = f"./{save_dir}/{language}_wikipedia_llama_100k.pt"
    torch.save(token_ids, fname)
    print("Saved tokens:", fname)
    print("Total tokens:", token_ids.numel())

for lang in ["zh", "en", "ja", "bn",  "sw", "ru",  "de", "es", "fr", "te", "th"]:
    tokenize_wiki(tokenizer, lang)
