#!/bin/bash
#SBATCH --job-name=rep_beh
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user/mzhang630/logs/rep_beh_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/rep_beh_%A_%a.err
#SBATCH --array=0-4

# Experiment 1: Representation-Behavior Dissociation
# Tests whether behavioral ToM improves with scale while MSI stays flat
#
# Array 0: Qwen2.5-0.5B-Instruct   (has RSA NPZ)
# Array 1: Qwen2.5-1.5B            (base, extract RSA inline)
# Array 2: Qwen2.5-1.5B-Instruct   (has RSA NPZ)
# Array 3: Qwen2.5-3B-Instruct     (has RSA NPZ)
# Array 4: Qwen2.5-14B-Instruct    (extract RSA inline, needs ~30GB VRAM)

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
RSA_DIR="${BASE}/results/cognitive_rsa"
BRAIN_RDM="${RSA_DIR}/brain_rdm.npz"
STIM="${BASE}/cognitive_stimuli/rsa/rsa_stimuli.jsonl"
OUT_DIR="${BASE}/results/rep_behavior_dissociation"
mkdir -p "${OUT_DIR}"

MODELS=(
    "/data/user/mzhang630/data/models_dl/Qwen2.5-0.5B-Instruct"
    "/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B"
    "/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
    "/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    "/data/user/mzhang630/data/models_dl/Qwen2.5-14B-Instruct"
)
SHORTS=(
    "Qwen2.5-0.5B-Instruct"
    "Qwen2.5-1.5B"
    "Qwen2.5-1.5B-Instruct"
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-14B-Instruct"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
SHORT="${SHORTS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Rep-Behavior Dissociation: ${SHORT} ==="
echo "Model: ${MODEL}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"

cd "${BASE}/src"
python rep_behavior_dissociation.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --rsa_dir "${RSA_DIR}" \
    --brain_rdm_path "${BRAIN_RDM}" \
    --stimuli_path "${STIM}" \
    --output_dir "${OUT_DIR}" \
    --peak_layer 26

echo "DONE: ${SHORT}"
