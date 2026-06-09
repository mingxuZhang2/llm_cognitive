#!/bin/bash
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=narr_xarch
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/narr_xarch_%A_%a.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/narr_xarch_%A_%a.err
#SBATCH --array=0-3

# Cross-architecture Narratives LLM extraction (the "same story to real brain & model"
# defense). One model per array task, all 12 target stories per task.
# This cluster (/hpc2hdd): GPU partition i64m1tga800u; conda base has torch 2.9 + cu128 +
# transformers 4.57. Source conda.sh directly (NOT ~/.bashrc). Use the offline HF cache.

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
echo "=== task $SLURM_ARRAY_TASK_ID : ${CFG[0]} ($(date)) ==="

cd /hpc2hdd/home/mzhang630/data/nature/experiments
python -u src/extract_narratives_xarch.py \
    --model_path "${CFG[1]}" \
    --model_short "${CFG[0]}" \
    --sentences_jsonl data/narratives/annotated_sentences.jsonl \
    --output_dir results/narratives_llm \
    --batch_size 8
echo "DONE ${CFG[0]} ($(date))"
