import os
import json
import random
import logging
import numpy as np
from datasets import load_dataset
from tqdm import tqdm 

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUTPUT_FILE = "language_subspace_probe_data.jsonl"
NUM_SAMPLES = 2000
SEED        = 42

LANGUAGE_MAP = {
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "bn": "bengali",
    "sw": "swahili",
    "ru": "russian",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "te": "telugu",
    "th": "thai",
}

random.seed(SEED)
np.random.seed(SEED)


def fetch_texts(lang_code: str, hf_name: str, n: int) -> list[str]:
    logging.info(f"[{lang_code}] Loading dataset info for '{hf_name}' ...")

    ds = load_dataset(
        "CohereLabs/aya_collection_language_split",
        hf_name,
        split="train",
        trust_remote_code=True,
    )

    total = len(ds)
    logging.info(f"[{lang_code}] Total rows available: {total:,}")

    n_actual = min(n, total)
    indices = np.random.choice(total, size=n_actual, replace=False).tolist()
    indices_sorted = sorted(indices)
    subset = ds.select(indices_sorted)

    texts = []
    for row in tqdm(subset, desc=f"[{lang_code}] Processing", leave=False):
        inp = (row.get("inputs") or "").strip()
        tgt = (row.get("targets") or "").strip()
        combined = f"{inp} {tgt}".strip() if tgt else inp
        if combined:
            texts.append(combined)

    random.shuffle(texts)
    logging.info(f"[{lang_code}] Collected {len(texts)} samples from {total:,} total rows")
    return texts


def main():
    lang_texts: dict[str, list[str]] = {}

    for code, hf_name in tqdm(LANGUAGE_MAP.items(), desc="Languages"):
        lang_texts[code] = fetch_texts(code, hf_name, NUM_SAMPLES)

    min_count = min(len(v) for v in lang_texts.values())
    if min_count < NUM_SAMPLES:
        logging.warning(
            f"Some languages have fewer than {NUM_SAMPLES} samples. "
            f"Aligning all to {min_count}."
        )
    else:
        logging.info(f"All languages have >= {NUM_SAMPLES} samples. Using {min_count}.")

    for code in lang_texts:
        lang_texts[code] = lang_texts[code][:min_count]

    written = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in tqdm(range(min_count), desc="Writing JSONL"):
            row = {code: lang_texts[code][i] for code in LANGUAGE_MAP}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    logging.info(f"Done. Wrote {written} lines to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    print("\n── Sample line (first row) ──")
    for code, text in sample.items():
        print(f"  [{code}] {text[:100]}")


if __name__ == "__main__":
    main()