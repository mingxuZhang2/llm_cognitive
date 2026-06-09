#!/bin/bash
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=narr_group
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/narr_group_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/narr_group_%j.err

# Group-level (denoised) Narratives brain-LLM RSA: average condition patterns across all
# subjects in Schaefer-400 space, build one clean group RDM, compare to 4 architectures,
# with a split-half-across-subjects noise ceiling. Loads real fMRI; big-mem CPU node; no GPU.

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=8

cd /hpc2hdd/home/mzhang630/data/nature/experiments
echo "===== narratives_group_rsa ($(date)) ====="
python -u src/narratives_group_rsa.py
echo "ALL DONE ($(date))"
