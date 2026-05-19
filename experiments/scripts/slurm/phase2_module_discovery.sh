#!/bin/bash
#SBATCH --job-name=funcatlas_discover
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/discover_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/discover_%A_%a.err
#SBATCH --array=0-3

# Phase 2: GPU-accelerated module discovery
# ~20sec per K per model on H100 (was 30min+ on CPU)
# Submit after Phase 1: sbatch phase2_module_discovery.sh

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer

MODELS=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "gemma-2-9b-it"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
ACT_DIR="/data/user/mzhang630/data/nature_exp/activations"
OUT_DIR="/data/user/mzhang630/data/nature_exp/modules"

echo "Running GPU module discovery for: ${MODEL}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python /data/user/mzhang630/data/nature_exp/src/module_discovery.py \
    --activation_path "${ACT_DIR}/${MODEL}_activations.npy" \
    --output_dir "${OUT_DIR}" \
    --K 5 8 10 15 20

echo "Done: ${MODEL}"
