#!/bin/bash
#SBATCH --job-name=specific_v2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/specific_v2_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/specific_v2_%A_%a.err
#SBATCH --array=0-3

# Specificity control on v2 atlas (one-vs-rest moral_selectivity, n=5000).
# Tests double dissociation: moral atlas should hurt norm_type more than
# neutral atlas does, AND neutral atlas should hurt neutral PPL more than
# moral atlas does.

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

OUTDIR="${BASE}/results/specificity_v2"
META_FILE="${BASE}/activations/${MODEL_SHORT}_cognitive_pilot_meta.json"
ATTR_V1="${BASE}/results/cognitive_pilot/${MODEL_SHORT}_multi_attribution.npz"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

cd "${SRC}"
python specificity_control_v2.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --attribution_npz_v2 "${ATTR_V1}" \
    --decomposition_jsonl "${STIM_DIR}/decomposition_stimuli.jsonl" \
    --meta_path "${META_FILE}" \
    --output_dir "${OUTDIR}"

echo "DONE: ${MODEL_SHORT}"
