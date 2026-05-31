#!/bin/bash
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=rsa_recompute
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/rsa_recompute_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/rsa_recompute_%j.err

# Recompute the CPU/no-GPU RSA analyses against the CORRECTED pure-Neurosynth
# brain_rdm.npz (HCP-ToM artifact fixed). Run on a 512GB compute node because the
# login node is memory-saturated and OOM-kills these (they load multiple ~500MB
# per_stim NPZ + float64 expansions + bootstrap/permutation).
#
# Already completed on login node (do NOT re-run here): rsa_deep_analysis,
# fix_all_holes, baseline_controls, permutation p.
# Remaining: scaling curve + noise ceiling; confirmatory (max-stat p, bootstrap CI).

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "===== rsa_scaling_analysis (corrected scaling curve + noise ceiling) ====="
python -u src/rsa_scaling_analysis.py

echo "===== confirmatory_rsa (max-stat p, bootstrap CI, cv-ablation) ====="
python -u src/confirmatory_rsa.py

echo "ALL DONE"
