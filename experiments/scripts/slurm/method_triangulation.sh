#!/bin/bash
#SBATCH --job-name=method_triang
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/method_triang_%j.out
#SBATCH --error=/data/user/mzhang630/logs/method_triang_%j.err

# Method Triangulation: Compare 3 attribution methods (G×A, Act-only, Grad-only)
# Single model (Qwen) on medium stimuli (50/cat, 400 total)

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

MODEL_SHORT="Qwen2.5-7B-Instruct"
MODEL_PATH="/data/user/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

OUTDIR="${BASE}/results/method_triangulation"
mkdir -p ${OUTDIR} /data/user/mzhang630/logs

echo "Method Triangulation: ${MODEL_SHORT}"
echo "Stimuli: stimuli_medium.jsonl (50/category, 400 total, balanced)"
echo "Methods: grad_x_act, act_only, grad_only"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

cd ${SRC}
python method_triangulation.py \
    --model_path "${MODEL_PATH}" \
    --model_short "${MODEL_SHORT}" \
    --meta_path "${BASE}/activations/${MODEL_SHORT}_meta.json" \
    --stimuli "${BASE}/stimuli/stimuli_medium.jsonl" \
    --output_dir "${OUTDIR}" \
    --n_ablate 5000

echo "DONE: ${MODEL_SHORT} method triangulation"
echo "Finished: $(date)"
