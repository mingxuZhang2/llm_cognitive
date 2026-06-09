#!/bin/bash
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=steer_ctrl
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/steer_ctrl_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/steer_ctrl_%j.err

# Control-direction steering: random + sentiment + PC1 controls.
# 3 controls x 30 prompts x 5 alphas = 450 responses (+ 4 extra random seeds = 750 for random).
# Total ~1050 generate calls. Expect ~2-3h on A800.
#
# This cluster (/hpc2hdd): GPU partition i64m1tga800u; conda base has torch + transformers.
# Source conda.sh directly (NOT ~/.bashrc).

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

MODEL_PATH="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
EXPDIR=/hpc2hdd/home/mzhang630/data/nature/experiments

mkdir -p $EXPDIR/logs

cd $EXPDIR

echo "=== Steering controls ($(date)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Step 1: Precompute PC1 direction from per-stim NPZ (CPU, ~1min).
# This avoids loading the ~444MB NPZ during the GPU job.
echo "--- Precomputing PC1 direction ---"
python -u src/steering_controls.py \
    --precompute_pc1_only \
    --model_short Qwen2.5-7B-Instruct \
    --peak_layer 27

# Step 2: Run all 3 controls (GPU).
echo "--- Running all controls ---"
python -u src/steering_controls.py \
    --model_path "$MODEL_PATH" \
    --model_short Qwen2.5-7B-Instruct \
    --peak_layer 27 \
    --controls random,sentiment,pc1

echo "=== DONE ($(date)) ==="
