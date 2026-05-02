#!/bin/bash
#SBATCH -J llama-random
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --output="./logs/lambda_0.05_fixed_seed_999.log"
#SBATCH --error="./logs/lambda_0.05_fixed_seed_999.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
RANDOM_SEED=999
export PYTHONHASHSEED=$RANDOM_SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8

unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml-training-env

REG_LAMBDA=0.05
OUTPUT_DIR="./outputs/output_lambda_fixed_seed_999_${REG_LAMBDA}"
mkdir -p "$OUTPUT_DIR"

python train.py \
    --model_path        "./llama3b" \
    --data_dir          "." \
    --val_file          "./data/math_validation.jsonl" \
    --subspace_path     "./lang_specific_space_last.pkl" \
    --output_dir        "$OUTPUT_DIR" \
    --lora_rank         64 \
    --lora_alpha        16 \
    --lora_dropout      0.1 \
    --reg_lambda        "$REG_LAMBDA" \
    --learning_rate     1e-4 \
    --warmup_ratio      0.03 \
    --per_device_batch_size 4 \
    --grad_accum_steps  32 \
    --max_seq_len       1024 \
    --num_epochs        40 \
    --eval_steps        150 \
    --patience          5 \
    --seed              "$RANDOM_SEED"

conda activate lmeval

python plot_training_curves.py \
    --output_dir "$OUTPUT_DIR" \
    --reg_lambda "$REG_LAMBDA" \
    --all_plots          
