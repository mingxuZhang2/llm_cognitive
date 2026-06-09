#!/bin/bash
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --job-name=moral_judge
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/moral_judgment_%A_%a.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/moral_judgment_%A_%a.err
#SBATCH --array=0-3

# Moral judgment steering test (Greene/Koenigs analog).
# One model per array task. 30 dilemmas x 9 alphas x 3 samples = 810 generations per model.
# This cluster (/hpc2hdd): GPU partition i64m1tga800u; conda base has torch + transformers.
# Source conda.sh directly (NOT ~/.bashrc).

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

HUB=/hpc2hdd/home/mzhang630/.cache/huggingface/hub
MODELS=(
  "Qwen2.5-7B-Instruct|$HUB/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  "Meta-Llama-3.1-8B-Instruct|$HUB/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ac6a82c495cbcdf8e379063f645bb928ff65be7f"
  "Mistral-7B-Instruct-v0.3|$HUB/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
  "gemma-2-9b-it|$HUB/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
)
IFS='|' read -ra CFG <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

EXPDIR=/hpc2hdd/home/mzhang630/data/nature/experiments
OUTDIR=$EXPDIR/results/moral_judgment

mkdir -p $OUTDIR $EXPDIR/logs $EXPDIR/figures

echo "=== task $SLURM_ARRAY_TASK_ID : $MODEL_SHORT ($(date)) ==="
echo "=== path: $MODEL_PATH ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a') ==="

cd $EXPDIR

python -u src/moral_judgment_test.py \
    --model_path "$MODEL_PATH" \
    --model_short "$MODEL_SHORT" \
    --dilemmas_path data/cognitive_stimuli/moral/moral_dilemmas.jsonl \
    --output_dir "$OUTDIR"

echo "DONE $MODEL_SHORT ($(date))"
