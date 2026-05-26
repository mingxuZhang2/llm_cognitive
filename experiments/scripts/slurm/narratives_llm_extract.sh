#!/bin/bash
#SBATCH --job-name=narr_llm
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/data/user/mzhang630/logs/narr_llm_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/narr_llm_%A_%a.err
#SBATCH --array=0-3

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
SENTS="${BASE}/narratives/annotated_sentences.jsonl"
OUTDIR="${BASE}/results/narratives_llm"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

MODELS=(
    "Qwen2.5-7B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct|/hpc2hdd/home/mzhang630/data/models_dl/Meta-Llama-3.1-8B-Instruct"
    "Mistral-7B-Instruct-v0.3|/hpc2hdd/home/mzhang630/data/models_dl/Mistral-7B-Instruct-v0.3"
    "gemma-2-9b-it|/hpc2hdd/home/mzhang630/data/models_dl/gemma-2-9b-it"
)

IFS='|' read -ra CFG <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

cd "${BASE}/src"
python extract_narratives_llm.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --sentences_jsonl "${SENTS}" \
    --story pieman \
    --output_dir "${OUTDIR}"

echo "DONE: ${MODEL_SHORT}"
