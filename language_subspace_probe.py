import os
import json
import logging
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = "/home/compiling-ganesh/24m0829/forgetting/x_transfer/llama3b"
DATASET_DIR = "/home/compiling-ganesh/24m0829/forgetting/x_transfer"
DATASET_FILE = "language_subspace_probe_data.jsonl"
LANGUAGE_LIST = ["en", "zh", "ja", "bn", "sw", "ru", "de", "es", "fr", "te", "th"]
SEED = 42
MAX_INPUT_LENGTH = 64
BATCH_SIZE = 50

def seed_all(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_model(model_path: str):
    # Load tokenizer and model from the local checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    logging.info(f"Model loaded from {model_path}")
    return model, tokenizer

def build_aligner(rank: int, lan_emb: dict):
    """Returns the basis of the language specific subspace"""

    # Compute one mean embedding per language.
    lan_mean_emb = {lan: np.mean(emb, axis=0) for lan, emb in lan_emb.items()}
    W = np.stack(list(lan_mean_emb.values())).T
    _, D = W.shape

    # Center the embedding matrix and estimate its low-rank structure.
    wc = W @ np.ones(D) / D
    u, s, vh = np.linalg.svd(W - wc.reshape(-1, 1) @ np.ones((1, D)))
    Ws = u[:, :rank]

    Gamma = vh.T[:, :rank] @ np.diag(s[:rank])
    best_fit_W = wc.reshape(-1, 1) @ np.ones((1, D)) + Ws @ Gamma.T
    wc_new = np.linalg.pinv(best_fit_W).T @ np.ones(D)
    wc_new /= (wc_new ** 2).sum()
    prod = best_fit_W - wc_new.reshape(-1, 1) @ np.ones((1, D))
    u2, _, _ = np.linalg.svd(prod)
    ws_new = u2[:, :rank]
    return wc_new, ws_new

class Probe:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.data_dir = DATASET_DIR

    def _load_data(self) -> dict:
        """
        Returns:
            tokenized: dict[str, BatchEncoding]
            {
            "en": {
                "input_ids": Tensor [N, 64],
                "attention_mask": Tensor [N, 64]
            },
            "zh": {...},
            ...
            }
        """
        filepath = os.path.join(self.data_dir, DATASET_FILE)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")
        lang_data: dict[str, list[str]] = {lang: [] for lang in LANGUAGE_LIST}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for lang in LANGUAGE_LIST:
                    text = row.get(lang, "").strip()
                    lang_data[lang].append(text)

        for lang, texts in lang_data.items():
            logging.info(f"[{lang}] {len(texts)} samples | example: {texts[0][:80]}")
        tokenized = {
            lang: self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LENGTH, # 64
            )
            for lang, texts in lang_data.items()
        }
        return tokenized

    def _get_hidden_embeddings(self, inputs) -> torch.Tensor:
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        total = input_ids.size(0)
        batch_size = min(BATCH_SIZE, total)
        num_batches = total // batch_size
        sent_embs = []
        for i in range(num_batches):
            b_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            b_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            logging.info(f"  Batch {i + 1}/{num_batches} (size {b_ids.size(0)})")
            with torch.no_grad():
                out = self.model(
                    input_ids=b_ids.to(self.model.device),
                    attention_mask=b_mask.to(self.model.device),
                    output_hidden_states=True,
                )
                hidden = out.hidden_states
            del out
            hidden = torch.stack(hidden[1:]) # Stack layers (skip embedding layer) → [L, B, T, H]
            sent = hidden[:, :, -1, :] # Take last token representation → [L, B, H]
            sent_embs.append(sent.detach().cpu())
            del hidden, sent
            torch.cuda.empty_cache()
        result = torch.cat(sent_embs, dim=1)
        logging.info(f"  Hidden states shape: {result.shape}")
        torch.cuda.empty_cache()
        return result

    def compute(self):
        lang_data = self._load_data()
        source_emb: dict[str, torch.Tensor] = {}
        for lang, data in tqdm(lang_data.items(), desc="Extracting embeddings"):
            logging.info(f"Computing embeddings for [{lang}]")
            source_emb[lang] = self._get_hidden_embeddings(data)
        """
            source_emb: dict[str, Tensor]
            {
            "en": Tensor [L, N, H],
            "zh": Tensor [L, N, H],
            ...
            }
            Each language maps to a tensor of hidden states:
            L = number of layers,
            N = number of texts,
            H = hidden size (embedding dimension).
        """
        num_layers = source_emb["en"].shape[0]
        rank = len(LANGUAGE_LIST) - 1
        preference_matrix = []
        for layer_idx in tqdm(range(num_layers), desc="Building aligners"):
            layer_emb = {lang: emb[layer_idx].numpy() for lang, emb in source_emb.items()}
            _, aligner = build_aligner(rank, layer_emb) # aligner shape: [H, rank] (language specific subspace basis)
            preference_matrix.append(torch.tensor(aligner.T))
        preference_matrix = torch.stack(preference_matrix, dim=0)
        final_path = os.path.join(self.data_dir, "lang_specific_space_last.pkl")
        with open(final_path, "wb") as f:
            pickle.dump(preference_matrix, f)
        logging.info(f"Subspace matrices saved to {final_path}.")

if __name__ == "__main__":
    seed_all(SEED)
    model, tokenizer = load_model(MODEL_PATH)
    probe = Probe(model=model, tokenizer=tokenizer)
    probe.compute()