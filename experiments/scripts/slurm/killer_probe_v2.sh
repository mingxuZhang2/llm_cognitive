#!/bin/bash
#SBATCH --job-name=probe_v2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/probe_v2_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/probe_v2_%A_%a.err
#SBATCH --array=0-1

# Experiment 3: Agent-State Factorization Probe v2
# Tests mental-state separability without verb signals
# Conditions: A (explicit), B (implicit), C (agent-indexed), D (recombination)
#
# Array 0: Qwen2.5-1.5B-Instruct
# Array 1: Qwen2.5-3B-Instruct

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
OUT_DIR="${BASE}/results/agent_state_v2"
mkdir -p "${OUT_DIR}"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
    SHORT="Qwen2.5-1.5B-Instruct"
else
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    SHORT="Qwen2.5-3B-Instruct"
fi

echo "=== Agent-State Probe v2: ${SHORT} ==="
cd "${BASE}/src"
python agent_state_probe_v2.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --output_dir "${OUT_DIR}" \
    --n_per_condition 80

echo "DONE: ${SHORT}"
