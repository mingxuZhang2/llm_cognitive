#!/bin/bash
#SBATCH --job-name=funcatlas_gbsplit
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user/mzhang630/logs/gbsplit_%A.out
#SBATCH --error=/data/user/mzhang630/logs/gbsplit_%A.err

# Gemma balanced split rerun — fix torch.compile recompilation limit

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

MODEL_SHORT="gemma-2-9b-it"
MODEL_PATH="/data/user/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"

mkdir -p ${BASE}/results/balanced_split /data/user/mzhang630/logs

echo "Gemma Balanced Split Fix: ${MODEL_SHORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE}"

cd ${SRC}
python accuracy_dissociation.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${BASE}/activations/${MODEL_SHORT}_meta.json" \
    --discovery_stimuli "${BASE}/stimuli/stimuli_with_answers_balanced_discovery.jsonl" \
    --validation_stimuli "${BASE}/stimuli/stimuli_with_answers_balanced_validation.jsonl" \
    --output_dir "${BASE}/results/balanced_split" \
    --n_ablate 5000

echo "DONE: ${MODEL_SHORT}"
