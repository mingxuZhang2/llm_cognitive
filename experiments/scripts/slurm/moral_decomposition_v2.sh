#!/bin/bash
#SBATCH --job-name=moral_decomp_v2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/moral_decomp_v2_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/moral_decomp_v2_%A_%a.err
#SBATCH --array=0-3

# Rerun ONLY Step 3 (moral_decomposition) with the patched script.
#   v2 fixes (from code review):
#     - apply_chat_template + assistant prefix (was: raw-text prompt; OOD for chat models)
#     - log-odds wrongness score logsumexp(P[5,6,7]) - logsumexp(P[1,2,3])
#       (was: expected rating, hit ceiling at 6.8/7 for most baselines)
#     - rating-token resolution via prompt+digit encoding (was: fragile SP heuristic)
#     - added moral_anti sanity-check ablation (bottom-k selectivity neurons)
#     - random control averaged over 3 seeds (was: single seed)
#
# Step 1 (activation_extraction, model dims) and Step 2 (multi_function_dissociation,
# attribution.npz) were completed in job 313585 — reused as-is.

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

OUTDIR="${BASE}/results/cognitive_pilot_v2"
ACT_DIR="${BASE}/activations"
ATTR_DIR="${BASE}/results/cognitive_pilot"
META_FILE="${ACT_DIR}/${MODEL_SHORT}_cognitive_pilot_meta.json"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

echo "============================================================"
echo "Moral Decomposition v2 (patched): ${MODEL_SHORT}"
echo "Reusing attribution from job 313585:"
echo "  ${ATTR_DIR}/${MODEL_SHORT}_multi_attribution.npz"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"

cd "${SRC}"

python moral_decomposition.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --attribution_npz "${ATTR_DIR}/${MODEL_SHORT}_multi_attribution.npz" \
    --decomposition_jsonl "${STIM_DIR}/decomposition_stimuli.jsonl" \
    --meta_path "${META_FILE}" \
    --output_dir "${OUTDIR}" \
    --n_ablate 5000

echo ""
echo "DONE: ${MODEL_SHORT}"
