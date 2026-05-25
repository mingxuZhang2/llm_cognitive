#!/bin/bash
#SBATCH --job-name=rsa_scale
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/rsa_scale_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/rsa_scale_%A_%a.err
#SBATCH --array=0-3

# Qwen2.5-Instruct scaling axis (skip 7B because it is already done).

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"
STIM="${BASE}/cognitive_stimuli/rsa/rsa_stimuli.jsonl"

# Each size: short_label | model_path
CONFIGS=(
    "Qwen2.5-0.5B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-0.5B-Instruct"
    "Qwen2.5-1.5B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
    "Qwen2.5-3B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    "Qwen2.5-14B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-14B-Instruct"
)
IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

OUTDIR="${BASE}/results/cognitive_rsa"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

cd "${SRC}"
python extract_rsa_activations_v2.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --stimuli_jsonl "${STIM}" \
    --output_dir "${OUTDIR}"

echo "DONE: ${MODEL_SHORT}"
