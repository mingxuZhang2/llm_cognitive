#!/bin/bash
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --job-name=reg_xarch
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/reg_xarch_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/reg_xarch_%j.err

# Cross-architecture regional Narratives brain-LLM RSA (proper, parcel-level, z-scored).
# Loads real fMRI (162 GB dataset, one BOLD per subject-story, freed after use) + nilearn
# Schaefer-400 (cached in ~/nilearn_data). Big-memory CPU node; no GPU. Brain side computed
# once and cached, then 4 architectures correlated against it.

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=8

cd /hpc2hdd/home/mzhang630/data/nature/experiments
echo "===== regional_rsa_xarch ($(date)) ====="
python -u src/regional_rsa_xarch.py
echo "ALL DONE ($(date))"
