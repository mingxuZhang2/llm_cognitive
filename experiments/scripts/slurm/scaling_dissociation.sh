#!/bin/bash
#SBATCH --job-name=funcatlas_scale
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user/mzhang630/logs/scale_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/scale_%A_%a.err
#SBATCH --array=0-8

# Scaling experiment: 8-way dissociation across model sizes
# 3 families × multiple sizes = 9 new models
# Uses medium stimuli (50/cat) for comparability with existing results

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"
MODEL_BASE="/data/user/mzhang630/data/nature_exp/models"

# Model configs: SHORT_NAME|PATH|FAMILY|PARAMS_B
CONFIGS=(
    "Qwen2.5-0.5B-Instruct|${MODEL_BASE}/Qwen2.5-0.5B-Instruct|qwen|0.5"
    "Qwen2.5-1.5B-Instruct|${MODEL_BASE}/Qwen2.5-1.5B-Instruct|qwen|1.5"
    "Qwen2.5-3B-Instruct|${MODEL_BASE}/Qwen2.5-3B-Instruct|qwen|3"
    "Qwen2.5-14B-Instruct|${MODEL_BASE}/Qwen2.5-14B-Instruct|qwen|14"
    "Llama-3.2-1B-Instruct|${MODEL_BASE}/Llama-3.2-1B-Instruct|llama|1"
    "Llama-3.2-3B-Instruct|${MODEL_BASE}/Llama-3.2-3B-Instruct|llama|3"
    "gemma-2-2b-it|${MODEL_BASE}/gemma-2-2b-it|gemma|2"
    "Qwen2.5-32B-Instruct|${MODEL_BASE}/Qwen2.5-32B-Instruct|qwen|32"
    "gemma-2-27b-it|${MODEL_BASE}/gemma-2-27b-it|gemma|27"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"
FAMILY="${CFG[2]}"
PARAMS="${CFG[3]}"

mkdir -p ${BASE}/results/scaling ${BASE}/activations /data/user/mzhang630/logs

echo "============================================================"
echo "Scaling Experiment: ${MODEL_SHORT} (${PARAMS}B, family=${FAMILY})"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

cd ${SRC}

# Step 1: Extract activations to get meta.json
ACT_FILE="${BASE}/activations/${MODEL_SHORT}_activations.npy"
META_FILE="${BASE}/activations/${MODEL_SHORT}_meta.json"
if [ ! -f "${META_FILE}" ]; then
    echo "[Step 1] Extracting activations for meta.json..."
    python activation_extraction.py \
        --model "${MODEL_PATH}" \
        --model_short "${MODEL_SHORT}" \
        --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
        --output_dir "${BASE}/activations" \
        --max_length 512 \
        --batch_size 16
else
    echo "[Step 1] Meta already exists, skipping activation extraction."
fi

# Step 2: Run 8-way dissociation
echo "[Step 2] Running 8-way dissociation..."
python multi_function_dissociation.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${META_FILE}" \
    --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
    --output_dir "${BASE}/results/scaling" \
    --n_ablate 5000

echo "DONE: ${MODEL_SHORT} (${PARAMS}B)"
