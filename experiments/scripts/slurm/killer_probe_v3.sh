#!/bin/bash
#SBATCH --job-name=probe_v3
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user/mzhang630/logs/probe_v3_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/probe_v3_%A_%a.err
#SBATCH --array=0-2

# Agent-state binding v3: counter-balanced + MLP probe + layer sweep
# Array 0: Qwen2.5-1.5B-Instruct
# Array 1: Qwen2.5-3B-Instruct
# Array 2: Meta-Llama-3.1-8B-Instruct (different architecture)

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
OUT_DIR="${BASE}/results/agent_state_v3"
mkdir -p "${OUT_DIR}"

case $SLURM_ARRAY_TASK_ID in
    0)
        MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
        SHORT="Qwen2.5-1.5B-Instruct"
        ;;
    1)
        MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
        SHORT="Qwen2.5-3B-Instruct"
        ;;
    2)
        MODEL="/data/user/mzhang630/data/models_dl/Meta-Llama-3.1-8B-Instruct"
        SHORT="Meta-Llama-3.1-8B-Instruct"
        ;;
esac

echo "=== Agent-State Binding v3: ${SHORT} ==="
cd "${BASE}/src"
python agent_state_probe_v3.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --output_dir "${OUT_DIR}" \
    --n_per_pair 30 \
    --layer_stride 4

echo "DONE: ${SHORT}"
