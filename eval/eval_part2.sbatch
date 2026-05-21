#!/bin/bash
#SBATCH -J eval_p2
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --output="eval_p2.log"
#SBATCH --error="eval_p2.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml-training-env

# Directories
MATH_DATA_DIR="/home/compiling-ganesh/24m0829/forgetting/x_transfer/eval/mgsm_data"
XNLI_DATA_DIR="/home/compiling-ganesh/24m0829/forgetting/x_transfer/eval/xnli_data"

MATH_ADAPTER_DIR="/home/compiling-ganesh/24m0829/forgetting/x_transfer/math_partial_outputs"
XNLI_ADAPTER_DIR="/home/compiling-ganesh/24m0829/forgetting/x_transfer/xnli_outputs"

# ---------------------------------------------------------
# Part 2: 13 Runs (7 on GPU 0, 6 on GPU 1)
# ---------------------------------------------------------

(
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.4" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.15" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.2" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.25" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.3" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.35" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
) &

(
    sleep 5
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.4" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.45" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.5" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_999_0.0" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_999_0.05" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_999_0.1" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
) &

wait
