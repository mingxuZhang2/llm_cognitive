#!/bin/bash
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --job-name=gen_rate
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/gen_rate_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/gen_rate_%j.err

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
export TORCHDYNAMO_DISABLE=1

cd /hpc2hdd/home/mzhang630/data/nature/experiments

MODEL_PATH="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

echo "=== Generate rating responses ($(date)) ==="

python -u src/generate_rating_responses.py \
    --model_path "$MODEL_PATH" \
    --model_short Qwen2.5-7B-Instruct \
    --peak_layer 27 \
    

echo "=== DONE ($(date)) ==="
