#!/bin/bash
#SBATCH -J llama-mask-grid
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --time=12:00:00

#SBATCH --output=/home/compiling-ganesh/24m0829/forgetting/x_transfer/mask_grid.log
#SBATCH --error=/home/compiling-ganesh/24m0829/forgetting/x_transfer/mask_grid.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

unset PYTHONPATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lmeval


TOP_RATES=(0.01 0.05 0.10 0.20)
BAR_RATIOS=(0.85 0.90 0.95)
FILTER_RATES=(0.85 0.90 0.95)
WITH_ENS=(0 1)

for TOP_RATE in "${TOP_RATES[@]}"; do
    for BAR_RATIO in "${BAR_RATIOS[@]}"; do
        for FILTER_RATE in "${FILTER_RATES[@]}"; do
            for WITH_EN in "${WITH_ENS[@]}"; do

                echo "======================================================"
                echo "TOP_RATE=$TOP_RATE"
                echo "BAR_RATIO=$BAR_RATIO"
                echo "FILTER_RATE=$FILTER_RATE"
                echo "WITH_EN=$WITH_EN"
                echo "======================================================"

                python -u create_masks.py \
                    --top_rate $TOP_RATE \
                    --activation_bar_ratio $BAR_RATIO \
                    --filter_rate $FILTER_RATE \
                    --with_en $WITH_EN

            done
        done
    done
done

echo "All mask generations complete."
