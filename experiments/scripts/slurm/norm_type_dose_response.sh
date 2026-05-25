#!/bin/bash
#SBATCH --job-name=norm_dose
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/data/user/mzhang630/logs/norm_dose_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/norm_dose_%A_%a.err
#SBATCH --array=0-3

# Dose-response on the norm_type contrast atlas.
# Per dose: 5 forward passes (1 top + 1 bot + 3 random seeds) over 100 stimuli
# 6 doses x 5 conditions x 100 stimuli ≈ 3000 fwd passes ≈ ~5 min/model

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

OUTDIR="${BASE}/results/norm_dose_response"
META_FILE="${BASE}/activations/${MODEL_SHORT}_cognitive_pilot_meta.json"
ATTR="${BASE}/results/contrast_pilot/${MODEL_SHORT}_contrast_attribution.npz"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

echo "============================================================"
echo "Norm-type Dose-Response: ${MODEL_SHORT}"
echo "============================================================"
cd "${SRC}"

python norm_type_dose_response.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --attribution_npz "${ATTR}" \
    --decomposition_jsonl "${STIM_DIR}/decomposition_stimuli.jsonl" \
    --meta_path "${META_FILE}" \
    --output_dir "${OUTDIR}"

echo "DONE: ${MODEL_SHORT}"
