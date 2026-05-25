#!/bin/bash
#SBATCH --job-name=cognitive_pilot
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user/mzhang630/logs/cognitive_pilot_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/cognitive_pilot_%A_%a.err
#SBATCH --array=0-3

# Cognitive Atlas v2 pilot:
#   Step 1: activation_extraction.py  -> meta.json (model dims) on the 250-item
#           cognitive pilot stimuli (5 categories: emotion, moral, tom, self,
#           neutral_control, 50 each)
#   Step 2: multi_function_dissociation.py -> 5x5 dissociation matrix
#           + per-category attribution.npz (selectivity for each cognitive domain)
#   Step 3: moral_decomposition.py -> the linchpin compositional test on the 400
#           paired decomposition stimuli (4 conditions x 50 pairs)
#
# Outputs:
#   results/cognitive_pilot/{model}_multi_dissociation.json
#   results/cognitive_pilot/{model}_multi_attribution.npz
#   results/cognitive_pilot/{model}_moral_decomposition.json

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"
STIM_DIR="${BASE}/cognitive_stimuli"

CONFIGS=(
    "Qwen2.5-7B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    "Meta-Llama-3.1-8B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ac6a82c495cbcdf8e379063f645bb928ff65be7f"
    "Mistral-7B-Instruct-v0.3|/data/user/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    "gemma-2-9b-it|/data/user/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

OUTDIR="${BASE}/results/cognitive_pilot"
ACT_DIR="${BASE}/activations"
mkdir -p "${OUTDIR}" "${ACT_DIR}" /data/user/mzhang630/logs

echo "============================================================"
echo "Cognitive Atlas v2 Pilot: ${MODEL_SHORT}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

cd "${SRC}"

# Step 1: Extract meta (n_layers, ffn_dim) if not already done
META_FILE="${ACT_DIR}/${MODEL_SHORT}_cognitive_pilot_meta.json"
if [ ! -f "${META_FILE}" ]; then
    echo "[Step 1] Extracting model metadata via activation_extraction..."
    python activation_extraction.py \
        --model "${MODEL_PATH}" \
        --model_short "${MODEL_SHORT}_cognitive_pilot" \
        --stimuli "${STIM_DIR}/stimuli_cognitive_pilot.jsonl" \
        --output_dir "${ACT_DIR}" \
        --max_length 512 \
        --batch_size 16
else
    echo "[Step 1] Meta already present, skipping."
fi

# Step 2: 5-way dissociation (emotion, moral, tom, self, neutral_control)
echo ""
echo "[Step 2] 5-way cognitive dissociation..."
python multi_function_dissociation.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${META_FILE}" \
    --stimuli "${STIM_DIR}/stimuli_cognitive_pilot.jsonl" \
    --output_dir "${OUTDIR}" \
    --n_ablate 5000

# Step 3: Moral decomposition (THE linchpin — does ablating moral neurons
#         selectively collapse intent/norm discrimination while preserving outcome?)
echo ""
echo "[Step 3] Moral decomposition compositional test..."
python moral_decomposition.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --attribution_npz "${OUTDIR}/${MODEL_SHORT}_multi_attribution.npz" \
    --decomposition_jsonl "${STIM_DIR}/decomposition_stimuli.jsonl" \
    --meta_path "${META_FILE}" \
    --output_dir "${OUTDIR}" \
    --n_ablate 5000

echo ""
echo "DONE: ${MODEL_SHORT}"
