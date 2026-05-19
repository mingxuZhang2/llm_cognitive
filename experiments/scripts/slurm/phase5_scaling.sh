#!/bin/bash
#SBATCH --job-name=funcatlas_scale
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/mzhang630/logs/scaling_%j.out
#SBATCH --error=/data/user/mzhang630/logs/scaling_%j.err

# Phase 5: Modularity scaling law across Pythia suite
# Submit after Phase 0: sbatch phase5_scaling.sh (can run parallel with Phases 1-4)

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HF_HOME="/data/user/mzhang630/data/nature_exp/models"
SRC="/hpc2hdd/home/mzhang630/data/nature/experiments/src"
BASE="/data/user/mzhang630/data/nature_exp"

cd ${SRC}

python scaling_law.py \
    --stimuli "${BASE}/stimuli/all_stimuli.jsonl" \
    --output_dir "${BASE}/results/scaling" \
    --K 10 \
    --n_samples 2000 \
    --models \
        EleutherAI/pythia-70m \
        EleutherAI/pythia-160m \
        EleutherAI/pythia-410m \
        EleutherAI/pythia-1b \
        EleutherAI/pythia-1.4b \
        EleutherAI/pythia-2.8b \
        EleutherAI/pythia-6.9b \
        EleutherAI/pythia-12b
