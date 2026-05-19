#!/bin/bash
#SBATCH --job-name=funcatlas_prune
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/mzhang630/logs/prune_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/prune_%A_%a.err
#SBATCH --array=0-5

# Phase 4: Atlas-guided pruning experiments
# 2 models × 3 protect scenarios = 6 jobs
# Submit after Phase 2: sbatch phase4_pruning.sh

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate funcatlas

export HF_HOME="/data/user/mzhang630/data/nature_exp/models"
BASE="/data/user/mzhang630/data/nature_exp"
SRC="/hpc2hdd/home/mzhang630/data/nature/experiments/src"

# Config: model_name model_short protect_id protect_name target_bench collateral_bench
CONFIGS=(
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|0|math|gsm8k math|humaneval mmlu hellaswag"
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|1|code|humaneval mbpp|gsm8k mmlu hellaswag"
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|4|language|mmlu hellaswag|gsm8k humaneval"
    "meta-llama/Meta-Llama-3-8B-Instruct|Meta-Llama-3-8B-Instruct|0|math|gsm8k math|humaneval mmlu hellaswag"
    "meta-llama/Meta-Llama-3-8B-Instruct|Meta-Llama-3-8B-Instruct|1|code|humaneval mbpp|gsm8k mmlu hellaswag"
    "meta-llama/Meta-Llama-3-8B-Instruct|Meta-Llama-3-8B-Instruct|4|language|mmlu hellaswag|gsm8k humaneval"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${CFG[0]}"
SHORT="${CFG[1]}"
PROTECT_ID="${CFG[2]}"
PROTECT_NAME="${CFG[3]}"
TARGET="${CFG[4]}"
COLLATERAL="${CFG[5]}"

K=10

echo "Model: ${MODEL}, Protect: ${PROTECT_NAME} (module ${PROTECT_ID})"

python ${SRC}/atlas_pruning.py \
    --model "${MODEL}" \
    --module_path "${BASE}/modules/${SHORT}_K${K}_modules.npz" \
    --meta_path "${BASE}/activations/${SHORT}_meta.json" \
    --activation_path "${BASE}/activations/${SHORT}_activations.npy" \
    --protect_module ${PROTECT_ID} \
    --protect_name "${PROTECT_NAME}" \
    --target_benchmarks ${TARGET} \
    --collateral_benchmarks ${COLLATERAL} \
    --sparsity 0.2 0.3 0.4 0.5 0.6 0.7 \
    --output "${BASE}/results/pruning/${SHORT}_protect_${PROTECT_NAME}.json"
