#!/bin/bash
#SBATCH --job-name=dose_response
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user/mzhang630/logs/dose_response_%j.out
#SBATCH --error=/data/user/mzhang630/logs/dose_response_%j.err

# Dose-response curves for Qwen2.5-7B-Instruct
# Ablation sizes: 500, 1000, 2000, 5000, 10000, 20000
# Functions: math, code
# Uses pre-computed selectivity from multi_attribution.npz

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

MODEL_SHORT="Qwen2.5-7B-Instruct"
MODEL_PATH="/data/user/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

OUTDIR="${BASE}/results/dose_response"
mkdir -p ${OUTDIR} /data/user/mzhang630/logs

echo "Dose-Response Curves: ${MODEL_SHORT}"
echo "Functions: math, code"
echo "Ablation sizes: 500 1000 2000 5000 10000 20000"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd ${SRC}
python dose_response.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${BASE}/activations/${MODEL_SHORT}_meta.json" \
    --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
    --attribution_path "${BASE}/results/scaled_multi/${MODEL_SHORT}_multi_attribution.npz" \
    --output_dir "${OUTDIR}" \
    --functions math code \
    --sizes 500 1000 2000 5000 10000 20000

echo "DONE: ${MODEL_SHORT} dose-response"
