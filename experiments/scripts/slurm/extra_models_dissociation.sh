#!/bin/bash
#SBATCH --job-name=funcatlas_extra
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user/mzhang630/logs/extra_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/extra_%A_%a.err
#SBATCH --array=0-1

# Extra models: base model (Llama-2-7b) + reasoning model (DeepSeek-R1)
# Addresses GPT Pro Priority 4: model axis expansion

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

CONFIGS=(
    "Llama-2-7b-hf|/data/user/mzhang630/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    "DeepSeek-R1-Distill-Llama-8B|/data/user/mzhang630/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

# First extract activations to get meta.json
mkdir -p ${BASE}/activations ${BASE}/results/extra_models /data/user/mzhang630/logs

echo "Extra Model Dissociation: ${MODEL_SHORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd ${SRC}

# Step 1: Extract activations (needed for meta.json with n_layers/ffn_dim)
ACT_FILE="${BASE}/activations/${MODEL_SHORT}_activations.npy"
if [ ! -f "${ACT_FILE}" ]; then
    echo "[Step 1] Extracting activations..."
    python activation_extraction.py \
        --model "${MODEL_PATH}" \
        --model_short "${MODEL_SHORT}" \
        --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
        --output_dir "${BASE}/activations" \
        --max_length 512 \
        --batch_size 32
else
    echo "[Step 1] Activations already exist, skipping."
fi

# Step 2: Run multi-function dissociation
echo "[Step 2] Running 8-way dissociation..."
python multi_function_dissociation.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${BASE}/activations/${MODEL_SHORT}_meta.json" \
    --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
    --output_dir "${BASE}/results/extra_models" \
    --n_ablate 5000

echo "DONE: ${MODEL_SHORT}"
