#!/bin/bash
#SBATCH -J eval
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --output="eval.log"
#SBATCH --error="eval.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml-training-env

ADAPTER_DIR="/home/compiling-ganesh/24m0829/forgetting/x_transfer/outputs" 
DATA_DIR="/home/compiling-ganesh/24m0829/forgetting/x_transfer/eval/data"

(
    sleep 5 
    CUDA_VISIBLE_DEVICES=0 python eval_mgsm.py --adapter_path "$ADAPTER_DIR/output_lambda_fixed_seed_999_0.0" --data_dir "$DATA_DIR" --device "cuda:0" --langs en zh ja bn sw ru de es fr te th
    CUDA_VISIBLE_DEVICES=1 python eval_mgsm.py --adapter_path "$ADAPTER_DIR/output_lambda_fixed_seed_999_0.05" --data_dir "$DATA_DIR" --device "cuda:0" --langs en zh ja bn sw ru de es fr te th
) &

(
    sleep 35
    CUDA_VISIBLE_DEVICES=2 python eval_mgsm.py --adapter_path "$ADAPTER_DIR/output_lambda_fixed_seed_1234_0.05" --data_dir "$DATA_DIR" --device "cuda:0" --langs en zh ja bn sw ru de es fr te th
    CUDA_VISIBLE_DEVICES=3 python eval_mgsm.py --adapter_path "$ADAPTER_DIR/output_lambda_fixed_seed_123_0.05" --data_dir "$DATA_DIR" --device "cuda:0" --langs en zh ja bn sw ru de es fr te th
) &
wait

wait
