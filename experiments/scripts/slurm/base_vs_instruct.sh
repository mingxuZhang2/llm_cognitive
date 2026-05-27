#!/bin/bash
#SBATCH --job-name=base_inst
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/base_inst_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/base_inst_%A_%a.err
#SBATCH --array=0-1

# Array 0: Qwen2.5-1.5B base — emotion geometry
# Array 1: Qwen2.5-1.5B base — RSA cognitive axis (pieman story)

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B"
SHORT="Qwen2.5-1.5B"

cd "${BASE}/src"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "=== Base model: Emotion geometry ==="
    mkdir -p "${BASE}/results/emotion_geometry"
    python emotion_geometry.py \
        --model_path "${MODEL}" \
        --model_short "${SHORT}" \
        --stimuli "${BASE}/cognitive_stimuli/emotion/goemotions_sample.jsonl" \
        --output_dir "${BASE}/results/emotion_geometry"
else
    echo "=== Base model: Narrative RSA extraction ==="
    mkdir -p "${BASE}/results/narratives_llm"
    python extract_narratives_llm.py \
        --model_path "${MODEL}" \
        --model_short "${SHORT}" \
        --sentences_jsonl "${BASE}/narratives/annotated_sentences.jsonl" \
        --story pieman \
        --output_dir "${BASE}/results/narratives_llm"
fi

echo "DONE: task $SLURM_ARRAY_TASK_ID"
