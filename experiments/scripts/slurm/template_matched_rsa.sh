#!/bin/bash
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=tm_rsa
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/tm_rsa_%A_%a.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/tm_rsa_%A_%a.err
#SBATCH --array=0-3

# Template-matched RSA: format-confound control.
#
# Step 1 (GPU, this script): extract per-stimulus activations from
#   rsa_stimuli_template_matched.jsonl for all 4 models.
# Step 2 (CPU, runs after all array tasks): compute RSA vs brain_rdm.npz.
#
# This cluster (/hpc2hdd): GPU partition i64m1tga800u; conda base has
# torch 2.9 + cu128 + transformers 4.57. Source conda.sh directly
# (NOT ~/.bashrc). Use the offline HF cache.

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

echo "=== task $SLURM_ARRAY_TASK_ID : $MODEL_SHORT ($(date)) ==="
echo "=== path: $MODEL_PATH ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXPDIR=/hpc2hdd/home/mzhang630/data/nature/experiments
STIMULI=$EXPDIR/data/cognitive_stimuli/rsa/rsa_stimuli_template_matched.jsonl
OUTDIR=$EXPDIR/results/cognitive_rsa

mkdir -p $OUTDIR $EXPDIR/logs

cd $EXPDIR

# ── Step 1: GPU extraction ──
# Reuse extract_rsa_activations_v2.py with template-matched stimuli.
# Output: {model}_tm_per_stim.npz (named with _tm_ to distinguish from original)
python -u src/extract_rsa_activations_v2.py \
    --model_path "$MODEL_PATH" \
    --model_short "${MODEL_SHORT}_tm" \
    --stimuli_jsonl "$STIMULI" \
    --output_dir "$OUTDIR"

echo "=== Extraction done for $MODEL_SHORT ($(date)) ==="

# ── Step 2: CPU RSA analysis (only run from task 0 after all extractions) ──
# To avoid race conditions, run template_matched_rsa.py manually after all
# 4 array tasks complete:
#   python -u src/template_matched_rsa.py
# Or submit a separate CPU job that depends on this array job.

echo "=== DONE $MODEL_SHORT ($(date)) ==="
echo "=== After all 4 models finish, run: python -u src/template_matched_rsa.py ==="
