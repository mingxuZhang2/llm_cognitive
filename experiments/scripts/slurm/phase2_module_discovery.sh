#!/bin/bash
#SBATCH --job-name=funcatlas_discover
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/mzhang630/logs/discover_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/discover_%A_%a.err
#SBATCH --array=0-3

# Phase 2: Module discovery (CPU-intensive, no GPU needed)
# Submit after Phase 1: sbatch phase2_module_discovery.sh

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer

MODELS=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3-8B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "gemma-2-9b-it"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
ACT_DIR="/data/user/mzhang630/data/nature_exp/activations"
OUT_DIR="/data/user/mzhang630/data/nature_exp/modules"

echo "Running module discovery for: ${MODEL}"

python /hpc2hdd/home/mzhang630/data/nature/experiments/src/module_discovery.py \
    --activation_path "${ACT_DIR}/${MODEL}_activations.npy" \
    --output_dir "${OUT_DIR}" \
    --K 5 10 15 20 25

echo "Done: ${MODEL}"
