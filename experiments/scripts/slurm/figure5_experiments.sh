#!/bin/bash
#SBATCH --job-name=fig5
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/data/user/mzhang630/logs/fig5_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/fig5_%A_%a.err
#SBATCH --array=0-3

# Array 0: Prospective prediction (Qwen 3B — held-out model)
# Array 1: Prospective prediction (Qwen 0.5B — held-out model)
# Array 2: Agent-state probe (Qwen 3B)
# Array 3: Agent-state probe (Qwen 1.5B)

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"

cd "${BASE}/src"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "=== Prospective Prediction: Qwen 3B ==="
    mkdir -p "${BASE}/results/prospective"
    python prospective_prediction.py \
        --model_path /data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct \
        --model_short Qwen2.5-3B-Instruct \
        --output_dir "${BASE}/results/prospective"

elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "=== Prospective Prediction: Qwen 0.5B ==="
    mkdir -p "${BASE}/results/prospective"
    python prospective_prediction.py \
        --model_path /data/user/mzhang630/data/models_dl/Qwen2.5-0.5B-Instruct \
        --model_short Qwen2.5-0.5B-Instruct \
        --output_dir "${BASE}/results/prospective"

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo "=== Agent-State Probe: Qwen 3B ==="
    mkdir -p "${BASE}/results/agent_state"
    python agent_state_probe.py \
        --model_path /data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct \
        --model_short Qwen2.5-3B-Instruct \
        --output_dir "${BASE}/results/agent_state"

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    echo "=== Agent-State Probe: Qwen 1.5B ==="
    mkdir -p "${BASE}/results/agent_state"
    python agent_state_probe.py \
        --model_path /data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct \
        --model_short Qwen2.5-1.5B-Instruct \
        --output_dir "${BASE}/results/agent_state"
fi

echo "DONE: task $SLURM_ARRAY_TASK_ID"
