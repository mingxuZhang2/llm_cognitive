#!/bin/bash
#SBATCH --job-name=pred_mode
#SBATCH --partition=i64m1tga800u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pred_mode_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

MODEL_PATH="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

echo "=== Prediction mode analysis ($(date)) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
python3 src/prediction_mode_analysis.py \
    --model_path "$MODEL_PATH" \
    --model_short "Qwen2.5-7B-Instruct" \
    --top_k 50
echo "=== DONE ($(date)) ==="
