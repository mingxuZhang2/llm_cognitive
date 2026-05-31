#!/bin/bash
#SBATCH --job-name=vlm_cmp
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/vlm_cmp_%j.out
#SBATCH --error=/data/user/mzhang630/logs/vlm_cmp_%j.err

# Extract RSA activations from Qwen2.5-VL-3B-Instruct using TEXT-ONLY input.
# Compare with Qwen2.5-3B-Instruct (text-only model, same size).
# Difference must come from multimodal training, not architecture.

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"

cd "${BASE}/src"
python extract_rsa_activations_v2.py \
    --model_path "/data/user/mzhang630/data/models_dl/Qwen2.5-VL-3B-Instruct" \
    --model_short "Qwen2.5-VL-3B-Instruct" \
    --stimuli_jsonl "${BASE}/cognitive_stimuli/rsa/rsa_stimuli.jsonl" \
    --output_dir "${BASE}/results/cognitive_rsa"

echo "DONE: Qwen2.5-VL-3B-Instruct"
