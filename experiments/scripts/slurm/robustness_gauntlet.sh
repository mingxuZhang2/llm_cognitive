#!/bin/bash
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=robustness_gauntlet
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/robustness_gauntlet_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/robustness_gauntlet_%j.err

# Robustness gauntlet: 4 anti-spurious-alignment tests (CPU only).
# Loads multiple ~100-580MB per-stim NPZ files with float64 expansion,
# so needs the 512GB compute node (login node can OOM).

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "===== Robustness gauntlet ====="
python -u src/robustness_gauntlet.py

echo "DONE (exit $?)"
