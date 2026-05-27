#!/bin/bash
#SBATCH --job-name=narr_ms
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/data/user/mzhang630/logs/narr_ms_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/narr_ms_%A_%a.err
#SBATCH --array=0-4

# Extract LLM activations for 5 additional stories (pieman already done)
# Use Qwen2.5-1.5B as representative model

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
MODEL_SHORT="Qwen2.5-1.5B-Instruct"
SENTS="${BASE}/narratives/annotated_sentences.jsonl"
OUTDIR="${BASE}/results/narratives_llm"
mkdir -p "${OUTDIR}"

STORIES=(notthefallintact black prettymouth shapessocial shapesphysical)
STORY="${STORIES[$SLURM_ARRAY_TASK_ID]}"

cd "${BASE}/src"
python extract_narratives_llm.py \
    --model_path "${MODEL}" \
    --model_short "${MODEL_SHORT}" \
    --sentences_jsonl "${SENTS}" \
    --story "${STORY}" \
    --output_dir "${OUTDIR}"

echo "DONE: ${STORY}"
