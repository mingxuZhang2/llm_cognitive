#!/bin/bash
#SBATCH --job-name=deep_exp
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/deep_exp_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/deep_exp_%A_%a.err
#SBATCH --array=0-1

# Array 0: Cognitive steering (Experiment A)
# Array 1: Brain-to-LLM transfer (Experiment B)

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
MODEL_SHORT="Qwen2.5-3B-Instruct"
OUTDIR="${BASE}/results/deep_experiments"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

cd "${BASE}/src"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "=== Experiment A: Cognitive Steering ==="
    python cognitive_steering.py \
        --model_path "${MODEL}" \
        --model_short "${MODEL_SHORT}" \
        --output_dir "${OUTDIR}"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "=== Experiment B: Brain-to-LLM Transfer ==="
    python brain_transfer.py \
        --model_path "${MODEL}" \
        --model_short "${MODEL_SHORT}" \
        --brain_rdm "${BASE}/results/cognitive_rsa/narratives_brain_rdm.npz" \
        --output_dir "${OUTDIR}"
fi

echo "DONE: task $SLURM_ARRAY_TASK_ID"
