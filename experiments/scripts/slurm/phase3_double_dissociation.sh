#!/bin/bash
#SBATCH --job-name=funcatlas_dissoc
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user/mzhang630/logs/dissoc_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/dissoc_%A_%a.err
#SBATCH --array=0-23

# Phase 3: Double dissociation experiments
# 4 models × 3 pairs × 2 directions = 24 experiments
# Submit after Phase 2: sbatch phase3_double_dissociation.sh

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
)
MODEL_SHORTS=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3-8B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "gemma-2-9b-it"
)

# Dissociation configs: module_id benchmarks
# These module IDs will need to be filled in after Phase 2 identifies which module = which function
# Placeholder: assume module 0=math, 1=code, 2=knowledge, 3=reasoning, 4=language
PAIRS=(
    # pair_name  module_id  benchmarks
    "math_ablate 0 gsm8k,math,humaneval,mbpp"
    "code_ablate 1 gsm8k,math,humaneval,mbpp"
    "knowledge_ablate 2 triviaqa,arc_challenge,logiqa"
    "reasoning_ablate 3 triviaqa,arc_challenge,logiqa"
    "language_ablate 4 mmlu_humanities,hellaswag,gsm8k,math"
    "math_ablate2 0 mmlu_humanities,hellaswag,gsm8k,math"
)

# Decode array index
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 6))
PAIR_IDX=$((SLURM_ARRAY_TASK_ID % 6))

MODEL=${MODELS[$MODEL_IDX]}
MODEL_SHORT=${MODEL_SHORTS[$MODEL_IDX]}
PAIR_INFO=(${PAIRS[$PAIR_IDX]})
PAIR_NAME=${PAIR_INFO[0]}
MODULE_ID=${PAIR_INFO[1]}
BENCHMARKS=${PAIR_INFO[2]}

K=10  # Use K=10 modules (adjust after Phase 2 analysis)
MODULE_PATH="/data/user/mzhang630/data/nature_exp/modules/${MODEL_SHORT}_K${K}_modules.npz"
META_PATH="/data/user/mzhang630/data/nature_exp/activations/${MODEL_SHORT}_meta.json"
OUTPUT="/data/user/mzhang630/data/nature_exp/results/dissociation/${MODEL_SHORT}_${PAIR_NAME}.json"

export HF_HOME="/data/user/mzhang630/data/nature_exp/models"

echo "Model: ${MODEL}"
echo "Ablating module: ${MODULE_ID} (${PAIR_NAME})"
echo "Benchmarks: ${BENCHMARKS}"

IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"

python /hpc2hdd/home/mzhang630/data/nature/experiments/src/double_dissociation.py \
    --model "${MODEL}" \
    --module_path "${MODULE_PATH}" \
    --meta_path "${META_PATH}" \
    --target_module ${MODULE_ID} \
    --benchmarks "${BENCH_ARRAY[@]}" \
    --output "${OUTPUT}"

echo "Done: ${MODEL} / ${PAIR_NAME}"
