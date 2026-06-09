#!/bin/bash
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --job-name=pythia_dev
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/pythia_dev_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/pythia_dev_%j.err

# Track emergence of brain-like cognitive structure across Pythia-2.8B training
# checkpoints. Downloads model checkpoints from HuggingFace (needs internet).
#
# This cluster (/hpc2hdd): GPU partition i64m1tga800u (A800 80GB).
# Source conda.sh directly (NOT ~/.bashrc — it early-returns non-interactively).
# ~20 checkpoints x ~10 min each = ~3-4 hours; wall limit 8h for safety.

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

# HuggingFace: use Chinese mirror, bypass broken system proxy
export HF_HOME=/hpc2hdd/home/mzhang630/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
unset HTTPS_PROXY HTTP_PROXY http_proxy https_proxy

EXPDIR=/hpc2hdd/home/mzhang630/data/nature/experiments
mkdir -p "$EXPDIR/logs" \
         "$EXPDIR/results/developmental_emergence" \
         "$EXPDIR/figures"

echo "=== Pythia developmental trajectory ($(date)) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Python: $(which python) ==="

cd "$EXPDIR"

python -u src/pythia_developmental.py \
    --batch_size 8 \
    --max_length 256 \
    --save_rdms

echo "=== DONE ($(date)) ==="
