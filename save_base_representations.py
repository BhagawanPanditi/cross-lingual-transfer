import os, json, random, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH  = "./llama3b"
TRAIN_FILE  = "./data/xnli/xnli_train.jsonl"
OUT_DIR     = "./"
MAX_SEQ_LEN = 1024
BATCH_SIZE  = 4
SEED        = 42

# Define the new output file names
OUT_JSONL = os.path.join(OUT_DIR, "xnli_train_modified.jsonl")
OUT_PT    = os.path.join(OUT_DIR, "xnli_base_representations.pt")

set_seed(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
with open("base_chat_template.jinja") as f:
    tokenizer.chat_template = f.read()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
model.eval()
device = next(model.parameters()).device

# ── load data ──────────────────────────────────────────────────────────
with open(TRAIN_FILE) as f:
    samples = [json.loads(l) for l in f]

random.shuffle(samples)

# ── 1. filter out long sequences FIRST ──────────────────────────────────
print("Filtering out sequences that exceed max_seq_len...")
valid_samples = []
dropped_count = 0

for s in tqdm(samples, desc="Checking lengths"):
    text = tokenizer.apply_chat_template(
        s["messages"], tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, truncation=False, add_special_tokens=True, return_attention_mask=False)
    length = len(enc["input_ids"])
    
    if length <= MAX_SEQ_LEN:
        valid_samples.append(s)
    else:
        dropped_count += 1

print(f"Kept {len(valid_samples)} samples. Dropped {dropped_count} samples (too long).")
samples = valid_samples

# ── 2. assign _pos AFTER filtering ──────────────────────────────────────
for i, s in enumerate(samples):
    s["_pos"] = i

os.makedirs(OUT_DIR, exist_ok=True)

# ── dataset definition ──────────────────────────────────────────────────
class MathDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        text = tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=False
        )
        return s["_pos"], text

def collate(batch):
    ids, texts = zip(*batch)
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN
    )
    return list(ids), enc

# ── 3. save .jsonl and generate .pt file ────────────────────────────────

# Save JSONL
with open(OUT_JSONL, "w") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"\nSaved {len(samples)} samples to {OUT_JSONL}.")

# Save PT representations
if os.path.exists(OUT_PT):
    print(f"Representation file already exists at {OUT_PT}, skipping PT generation.")
else:
    loader = DataLoader(
        MathDataset(samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate
    )

    all_ids, all_vecs, all_last_indices, all_input_ids = [], [], [], []

    for pos_ids, enc in tqdm(loader, desc="Generating Representations"):
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        last_indices = attention_mask.sum(dim=1) - 1  # (B,)

        # only the final layer: (B, T, D)
        last_hidden = out.hidden_states[-1]

        # gather last real token from final layer: (B, D)
        B = input_ids.size(0)
        gathered = last_hidden[torch.arange(B, device=device), last_indices]
        gathered = gathered.to(torch.float16).cpu()

        # Extract unpadded input IDs
        for i in range(B):
            seq_len = last_indices[i] + 1
            all_input_ids.append(input_ids[i, :seq_len].cpu())

        all_ids.extend(pos_ids)
        all_vecs.append(gathered)
        all_last_indices.extend(last_indices.cpu().tolist())
        del out, last_hidden, gathered
        torch.cuda.empty_cache()

    all_vecs = torch.cat(all_vecs, dim=0)   # (N, D)
    all_ids  = torch.tensor(all_ids, dtype=torch.int32)
    all_last_indices = torch.tensor(all_last_indices, dtype=torch.int32)

    # Save the input_ids alongside everything else
    torch.save({
        "ids": all_ids, 
        "vecs": all_vecs, 
        "last_indices": all_last_indices,
        "input_ids": all_input_ids
    }, OUT_PT)
    
    print(f"Representations saved to {OUT_PT}. Shape: {all_vecs.shape}")
