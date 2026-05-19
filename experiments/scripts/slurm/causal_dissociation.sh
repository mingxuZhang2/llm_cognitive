#!/bin/bash
#SBATCH --job-name=funcatlas_causal
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/causal_%j.out
#SBATCH --error=/data/user/mzhang630/logs/causal_%j.err

# Causal attribution + targeted dissociation
# Uses gradient x activation to find causally important neurons per task
# Then ablates top-N most selective neurons (500, 1000, 2000, 5000)
# Much more surgical than IterD module ablation (15% of network)

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

MODEL_SHORT="Qwen2.5-7B-Instruct"
MODEL_PATH="/data/user/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

mkdir -p ${BASE}/results/causal /data/user/mzhang630/logs

echo "Causal Attribution + Targeted Dissociation: ${MODEL_SHORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Stimuli: pilot set (120 balanced samples)"

cd ${SRC}
python causal_attribution.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${BASE}/activations/${MODEL_SHORT}_meta.json" \
    --stimuli "${BASE}/stimuli/stimuli_pilot.jsonl" \
    --output_dir "${BASE}/results/causal" \
    --ablation_sizes 500 1000 2000 5000

echo "DONE: ${MODEL_SHORT}"
