import torch
import gc, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import MethodType
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
local_model_dir = "./llama3b"
save_dir = "./activation_counts"
os.makedirs(save_dir, exist_ok=True)

print(f"Loading model on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    local_model_dir,
    local_files_only=True,
)
model.to(device)

num_layers = model.model.config.num_hidden_layers
intermediate_size = model.model.layers[0].mlp.gate_proj.out_features
act_fn = model.model.layers[0].mlp.act_fn

activated_neuron_counts = torch.zeros(
    num_layers, intermediate_size, dtype=torch.int32
).to(device)

print(f"Layers: {num_layers}, MLP size: {intermediate_size}")

def make_forward(idx):
    def custom_forward(self, x):
        activated_gate_proj = self.act_fn(self.gate_proj(x))
        up_proj = self.up_proj(x)
        activated_neuron_counts[idx] += (activated_gate_proj > 0).sum(dim=(0, 1)).to(torch.int32)
        down_proj = self.down_proj(activated_gate_proj * up_proj)
        return down_proj
    return custom_forward

for layer_idx in range(num_layers):
    mlp = model.model.layers[layer_idx].mlp
    mlp.forward = MethodType(make_forward(layer_idx), mlp)

for language in ["zh", "en", "ja", "bn",  "sw", "ru",  "de", "es", "fr", "te", "th"]:
    token_file = f"./wiki_language_tokens/{language}_wikipedia_llama_100k.pt"
    print(f"Processing language: {language}")
    print("Loading tokens:", token_file)
    
    activated_neuron_counts.zero_()
    
    token_tensor = torch.load(token_file, map_location="cpu")

    max_length = min(4096, model.config.max_position_embeddings)
    sz = min(token_tensor.size(0), 99_999_744)
    sz = sz // max_length * max_length

    input_ids = token_tensor[:sz].reshape(-1, max_length).to(device)

    batch_size = 2
    num_batches = input_ids.size(0) // batch_size

    print(f"Batches: {num_batches}, Seq len: {max_length}")

    with torch.inference_mode():
        for i in tqdm(range(num_batches), desc=language):
            batch = input_ids[i*batch_size:(i+1)*batch_size]
            _ = model(batch)

    del token_tensor
    del input_ids
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(
        {
            "activated_neuron_counts": activated_neuron_counts.cpu(),
            "num_tokens": sz
        },
        f"{save_dir}/llama3_activation_counts_{language}.pt"
    )

print("DONE.")