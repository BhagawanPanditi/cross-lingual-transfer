#!/bin/bash
#SBATCH -J eval_p1
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --output="eval_p1.log"
#SBATCH --error="eval_p1.log"

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
# Part 1: 14 Runs (7 on GPU 0, 7 on GPU 1)
# ---------------------------------------------------------

(
    # GPU 0: 3 Math + 4 XNLI
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_0_0.0" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_0_0.05" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_0_0.1" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_1234_0.0" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=0 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_1234_0.05" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
) &i

(
    sleep 5
    # GPU 1: 7 XNLI
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_123_0.0" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_123_0.05" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_123_0.1" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.0" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.05" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
    CUDA_VISIBLE_DEVICES=1 python eval_xnli.py --adapter_path "$XNLI_ADAPTER_DIR/output_lambda_fixed_seed_42_0.1" --data_dir "$XNLI_DATA_DIR" --device "cuda:0" --langs all
) &

wait
