#!/bin/bash
#SBATCH --job-name=emo_base
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/emo_base_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/emo_base_%A_%a.err
#SBATCH --array=0-1

# Array 0: Emotion geometry with Qwen2.5-1.5B-Instruct
# Array 1: Emotion geometry with Qwen2.5-1.5B (BASE model, needs download)
# For now: run instruct on array 0, and 3B-Instruct on array 1 for scale comparison

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
STIM="${BASE}/cognitive_stimuli/emotion/goemotions_sample.jsonl"
OUTDIR="${BASE}/results/emotion_geometry"
mkdir -p "${OUTDIR}"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
    SHORT="Qwen2.5-1.5B-Instruct"
else
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    SHORT="Qwen2.5-3B-Instruct"
fi

cd "${BASE}/src"
python emotion_geometry.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --stimuli "${STIM}" \
    --output_dir "${OUTDIR}"

echo "DONE: ${SHORT}"
