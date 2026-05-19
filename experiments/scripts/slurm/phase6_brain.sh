#!/bin/bash
#SBATCH --job-name=funcatlas_brain
#SBATCH --partition=acd_u
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/brain_%j.out
#SBATCH --error=/data/user/mzhang630/logs/brain_%j.err

# Phase 6: Brain network comparison (CPU only)
# Submit after Phase 2: sbatch phase6_brain.sh

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer

SRC="/hpc2hdd/home/mzhang630/data/nature/experiments/src"
BASE="/data/user/mzhang630/data/nature_exp"

python ${SRC}/brain_comparison.py \
    --models \
        Qwen2.5-7B-Instruct \
        Meta-Llama-3-8B-Instruct \
        Mistral-7B-Instruct-v0.3 \
        gemma-2-9b-it \
    --activation_dir "${BASE}/activations" \
    --module_dir "${BASE}/modules" \
    --K 10 \
    --output "${BASE}/results/brain_comparison.json"
