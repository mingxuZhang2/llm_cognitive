#!/bin/bash
#SBATCH --job-name=funcatlas_balanced
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user/mzhang630/logs/balanced_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/balanced_%A_%a.err
#SBATCH --array=0-3

# Full balanced multi-dissociation: 152 samples/category, 1216 total
# Maximum statistical power for the final paper results

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

CONFIGS=(
    "Qwen2.5-7B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    "Meta-Llama-3.1-8B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ac6a82c495cbcdf8e379063f645bb928ff65be7f"
    "Mistral-7B-Instruct-v0.3|/data/user/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    "gemma-2-9b-it|/data/user/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

mkdir -p ${BASE}/results/balanced_multi /data/user/mzhang630/logs

echo "Balanced Multi-Function Dissociation: ${MODEL_SHORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Stimuli: balanced set (152/category, 1216 total)"

cd ${SRC}
python multi_function_dissociation.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${BASE}/activations/${MODEL_SHORT}_meta.json" \
    --stimuli "${BASE}/stimuli/stimuli_balanced.jsonl" \
    --output_dir "${BASE}/results/balanced_multi" \
    --n_ablate 5000

echo "DONE: ${MODEL_SHORT}"
