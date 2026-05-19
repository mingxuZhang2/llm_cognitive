#!/bin/bash
#SBATCH --job-name=dose_all
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/dose_all_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/dose_all_%A_%a.err
#SBATCH --array=0-2

# Dose-response for remaining 3 models (Qwen already done)

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

CONFIGS=(
    "Meta-Llama-3.1-8B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ac6a82c495cbcdf8e379063f645bb928ff65be7f"
    "Mistral-7B-Instruct-v0.3|/data/user/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    "gemma-2-9b-it|/data/user/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

mkdir -p ${BASE}/results/dose_response /data/user/mzhang630/logs

echo "Dose-Response: ${MODEL_SHORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd ${SRC}
python dose_response.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${BASE}/activations/${MODEL_SHORT}_meta.json" \
    --attribution_path "${BASE}/results/scaled_multi/${MODEL_SHORT}_multi_attribution.npz" \
    --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
    --output_dir "${BASE}/results/dose_response" \
    --functions math code \
    --ablation_sizes 500 1000 2000 5000 10000 20000

echo "DONE: ${MODEL_SHORT}"
