import os
import json
import argparse
from datasets import load_dataset

LABEL_MAP = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

LANGS = ["de", "en", "es", "fr", "ru", "sw", "th", "zh"]

INSTRUCTION = (
    "Determine the relationship between the premise and the hypothesis. "
    "Respond with only one word: Entailment, Contradiction, or Neutral."
)

def format_example(premise, hypothesis, label_int):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{INSTRUCTION}\n\nPremise: {premise}\nHypothesis: {hypothesis}"
            }
        ],
        "answer": LABEL_MAP[label_int]
    }

def write_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples):,} → {path}")

def main():
    parser = argparse.ArgumentParser()
    # Changed data_dir default value here:
    parser.add_argument("--data_dir", type=str, default="./xnli_data")
    args = parser.parse_args()

    # ----------------------------------------------------------
    # TEST: All languages (English test included as baseline)
    # ----------------------------------------------------------
    for lang in LANGS:
        print(f"\n[{lang}] Downloading test...")
        try:
            ds = load_dataset("xnli", lang)
            write_jsonl(
                [format_example(r["premise"], r["hypothesis"], r["label"]) for r in ds["test"]],
                os.path.join(args.data_dir, f"xnli_{lang}_test.jsonl")
            )
        except Exception as e:
            print(f"  Failed for {lang}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
