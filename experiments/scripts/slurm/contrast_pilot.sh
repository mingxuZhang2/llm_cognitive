#!/bin/bash
#SBATCH --job-name=contrast_pilot
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/contrast_pilot_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/contrast_pilot_%A_%a.err
#SBATCH --array=0-3

# Contrast-based attribution + 4x4 decomposition dissociation.
# Path A from the v2 review: instead of "moral vs other 4 domains" selectivity
# (which captures topic features of moral text), compute attribution as the pair
# contrast within each decomposition condition. Per condition, accumulate
# |grad×act|(a) - |grad×act|(b) across 50 pairs. Top-k positive = neurons
# selectively engaged for the a-side of that condition.
#
# Step 1: contrast_attribution.py -> {model}_contrast_attribution.npz
#         (uses the same 400 decomposition stimuli, no extra stimuli needed)
# Step 2: contrast_decomposition.py -> 4x4 ablation dissociation +
#         anti-ablation sanity check + 3-seed random noise floor

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

OUTDIR="${BASE}/results/contrast_pilot"
ACT_DIR="${BASE}/activations"
META_FILE="${ACT_DIR}/${MODEL_SHORT}_cognitive_pilot_meta.json"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

echo "============================================================"
echo "Contrast Pilot: ${MODEL_SHORT}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

cd "${SRC}"

echo "[Step 1] Contrast attribution from decomposition stimulus pairs..."
python contrast_attribution.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --decomposition_jsonl "${STIM_DIR}/decomposition_stimuli.jsonl" \
    --meta_path "${META_FILE}" \
    --output_dir "${OUTDIR}"

echo ""
echo "[Step 2] 4x4 contrast decomposition dissociation..."
python contrast_decomposition.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --attribution_npz "${OUTDIR}/${MODEL_SHORT}_contrast_attribution.npz" \
    --decomposition_jsonl "${STIM_DIR}/decomposition_stimuli.jsonl" \
    --meta_path "${META_FILE}" \
    --output_dir "${OUTDIR}" \
    --n_ablate 5000

echo ""
echo "DONE: ${MODEL_SHORT}"
