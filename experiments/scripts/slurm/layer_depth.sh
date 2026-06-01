#!/bin/bash
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=layer_depth
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/layer_depth_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/layer_depth_%j.err

# Direction D / cortical-gradient test: per-layer brain-LLM alignment profile.
# Loads the large *_rsa_v2_per_stim.npz (per-layer hidden states) -> needs a
# big-memory CPU node; OOM-kills on the login node. No GPU.
# Source conda.sh directly (NOT ~/.bashrc, which early-returns non-interactively).

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments
echo "===== layer_depth_analysis ($(date)) ====="
python -u src/layer_depth_analysis.py
echo "ALL DONE ($(date))"
