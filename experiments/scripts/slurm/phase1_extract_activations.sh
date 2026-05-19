#!/bin/bash
#SBATCH --job-name=funcatlas_extract
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/mzhang630/logs/extract_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/extract_%A_%a.err
#SBATCH --array=0-3

# Phase 1: Activation extraction for 4 models (array job)
# Submit: sbatch phase1_extract_activations.sh

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
STIMULI="/data/user/mzhang630/data/nature_exp/stimuli/all_stimuli.jsonl"
OUTPUT="/data/user/mzhang630/data/nature_exp/activations"
CACHE="/data/user/mzhang630/data/nature_exp/models"

export HF_HOME=${CACHE}
export TRANSFORMERS_CACHE=${CACHE}

echo "Extracting activations for: ${MODEL}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python /hpc2hdd/home/mzhang630/data/nature/experiments/src/activation_extraction.py \
    --model "${MODEL}" \
    --stimuli "${STIMULI}" \
    --output_dir "${OUTPUT}" \
    --max_length 512 \
    --batch_size 32

echo "Done: ${MODEL}"
