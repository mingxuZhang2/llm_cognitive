#!/bin/bash
#SBATCH --job-name=prosp_v2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/prosp_v2_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/prosp_v2_%A_%a.err
#SBATCH --array=0-1

# Experiment 2: Scaled Prospective Prediction (v2)
# Brain-derived MCS predicts held-out social reasoning failures
# 50 items/pair, 7 high-MCS + 7 low-MCS pairs
#
# Array 0: Qwen2.5-3B-Instruct
# Array 1: Qwen2.5-14B-Instruct

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
RSA_DIR="${BASE}/results/cognitive_rsa"
BRAIN_RDM="${RSA_DIR}/brain_rdm.npz"
OUT_DIR="${BASE}/results/prospective_v2"
mkdir -p "${OUT_DIR}"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    SHORT="Qwen2.5-3B-Instruct"
else
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-14B-Instruct"
    SHORT="Qwen2.5-14B-Instruct"
fi

echo "=== Prospective Prediction v2: ${SHORT} ==="
cd "${BASE}/src"
python prospective_prediction_v2.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --rsa_dir "${RSA_DIR}" \
    --brain_rdm_path "${BRAIN_RDM}" \
    --output_dir "${OUT_DIR}" \
    --peak_layer 26 \
    --n_pairs 7 \
    --items_per_pair 50

echo "DONE: ${SHORT}"
