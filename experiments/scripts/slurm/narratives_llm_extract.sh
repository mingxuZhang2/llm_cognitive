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
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
SENTS="${BASE}/narratives/annotated_sentences.jsonl"
OUTDIR="${BASE}/results/narratives_llm"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

MODELS=(
    "Qwen2.5-0.5B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-0.5B-Instruct"
    "Qwen2.5-1.5B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
    "Qwen2.5-3B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    "Qwen2.5-14B-Instruct|/data/user/mzhang630/data/models_dl/Qwen2.5-14B-Instruct"
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
