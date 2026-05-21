# Language Subspace Regularization for Cross-Lingual Transfer

A method for fine-tuning multilingual LLMs on a target task while preserving cross-lingual transfer ability. The core idea: identify the **language-specific subspace** of the model's representation space, then regularize training to keep the fine-tuned model's representations anchored in that subspace — preventing the model from "forgetting" how to generalize across languages.

---

## How It Works

Fine-tuning a multilingual model on an English task tends to degrade its cross-lingual performance. This method combats that by:

1. **Building a language subspace** from multilingual text, capturing the directions in representation space that encode language identity.
2. **Caching base representations** of the training data before fine-tuning begins.
3. **Regularizing during training** by penalizing drift in the language subspace component of each sample's representation — keeping cross-lingual structure intact while still learning the task.

---

## Setup

```bash
conda create -n ml-training-env python=3.10
conda activate ml-training-env
pip install torch transformers peft datasets tqdm numpy matplotlib seaborn
```

Download the base model (requires a HuggingFace token with Llama access):

```bash
# Edit download_base_model.py to set your HF_TOKEN
python download_base_model.py
# Saves to ./llama3b
```

---

## Pipeline

### Step 1 — Build the Language Subspace

**Collect multilingual probe data** from the Aya collection (11 languages):

```bash
python create_language_subspace_probe_data.py
# Output: language_subspace_probe_data.jsonl
```

**Fit the subspace** by extracting hidden states from the base model and running SVD to find the language-discriminative directions:

```bash
python language_subspace_probe.py
# Output: lang_specific_space_last.pkl
#         (shape: [num_layers, rank, hidden_size] — one subspace basis per layer)
```

This uses the **last-layer** hidden states of the base model. The resulting `.pkl` file encodes the subspace basis that will be used as the regularization target during training.

---

### Step 2 — Cache Base Representations

Before any fine-tuning, run the base model over the training set and save the final-layer hidden states for each sample. These are used as the regularization anchor.

```bash
python save_base_representations.py
# Reads:  ./data/xnli/xnli_train.jsonl   (or your task's training file)
# Output: xnli_train_modified.jsonl       (filtered, with _pos indices)
#         xnli_base_representations.pt    (cached last-token vectors)
```

---

### Step 3 — Fine-tune with Subspace Regularization

Train a LoRA adapter on the task. The loss is:

```
L_total = L_task + λ * L_reg
```

where `L_reg` is the MSE between the **subspace projection** of the current model's representations and the cached base representations.

Adjust the arguments in the train.sbatch file and submit the job.
```bash
sbatch train.sbatch
```

Key arguments:

| Argument | Description |
|---|---|
| `--reg_lambda` | Regularization strength (0 = standard LoRA, higher = stronger cross-lingual preservation) |
| `--subspace_path` | Path to `.pkl` file from Step 1 |

Training saves the best checkpoint (by validation loss) to `--output_dir`, along with `train_log.csv` and `val_log.csv`.

---

### Step 4 — Evaluate on MGSM

**Download the MGSM benchmark** (multilingual math word problems, 11 languages):

```bash
python eval/download_mgsm.py
# Output: eval/data/mgsm_{lang}_test.jsonl  for each language
```

**Run evaluation:**

```bash
python eval/eval_mgsm.py \
    --adapter_path ./outputs/run_lambda_0.3 \
    --data_dir     eval/data \
    --batch_size   64
```

**Aggregate results across all adapter runs:**

```bash
python eval/mgsm_summary.py
# Output: mgsm_results/lambda_comparison.csv
#         (rows = languages, columns = λ values)
```

---

## Repository Structure

```
.
├── create_language_subspace_probe_data.py  # Collect multilingual probe data
├── language_subspace_probe.py              # Fit language subspace via SVD
├── save_base_representations.py            # Cache base model hidden states
├── train.py                                # Fine-tuning with subspace regularization
├── train.sbatch                            # SLURM job script for training
├── base_chat_template.jinja                # Chat template used during training/eval
│
├── eval/
│   ├── download_mgsm.py                    # Download MGSM benchmark
│   ├── eval_mgsm.py                        # Run MGSM evaluation
│   └── mgsm_summary.py                     # Aggregate results → CSV
│
└── data/
    └── math/                               # Task-specific train/val data
```

---

## Languages

The subspace is built from 11 languages: English (`en`), Chinese (`zh`), Japanese (`ja`), Bengali (`bn`), Swahili (`sw`), Russian (`ru`), German (`de`), Spanish (`es`), French (`fr`), Telugu (`te`), Thai (`th`).


