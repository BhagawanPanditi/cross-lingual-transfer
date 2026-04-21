#!/bin/bash
#SBATCH -J llama-random
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --time=6:00:00
#SBATCH --output=/home/compiling-ganesh/24m0829/forgetting/x_transfer/create_masks.log
#SBATCH --error=/home/compiling-ganesh/24m0829/forgetting/x_transfer/create_masks.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
unset PYTHONPATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lmeval

python -u create_masks.py