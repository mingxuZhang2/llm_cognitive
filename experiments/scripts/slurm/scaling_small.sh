#!/bin/bash
#SBATCH --job-name=funcatlas_scale
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user/mzhang630/logs/scale_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/scale_%A_%a.err
#SBATCH --array=0-2

# Scaling experiment: small models (0.5B, 1.5B, 3B)
# 8-way dissociation on medium stimuli (50/cat)

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"
MODEL_BASE="/data/user/mzhang630/data/nature_exp/models"

CONFIGS=(
    "Qwen2.5-0.5B-Instruct|${MODEL_BASE}/Qwen2.5-0.5B-Instruct"
    "Qwen2.5-1.5B-Instruct|${MODEL_BASE}/Qwen2.5-1.5B-Instruct"
    "Qwen2.5-3B-Instruct|${MODEL_BASE}/Qwen2.5-3B-Instruct"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

mkdir -p ${BASE}/results/scaling ${BASE}/activations /data/user/mzhang630/logs

echo "============================================================"
echo "Scaling Experiment: ${MODEL_SHORT}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

cd ${SRC}

# Step 1: Extract activations for meta.json
META_FILE="${BASE}/activations/${MODEL_SHORT}_meta.json"
if [ ! -f "${META_FILE}" ]; then
    echo "[Step 1] Extracting activations for meta.json..."
    python activation_extraction.py \
        --model "${MODEL_PATH}" \
        --model_short "${MODEL_SHORT}" \
        --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
        --output_dir "${BASE}/activations" \
        --max_length 512 \
        --batch_size 32
else
    echo "[Step 1] Meta already exists, skipping."
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

echo "DONE: ${MODEL_SHORT}"
