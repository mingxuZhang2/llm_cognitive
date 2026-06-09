#!/bin/bash
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --job-name=subspace_dd
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/subspace_dd_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/subspace_dd_%j.err

# Subspace double-dissociation: project out emotion/social PCA subspaces from
# per-stimulus activations, measure selective collapse of brain-LLM alignment.
#
# CPU-only (no GPU). Loads large per-stim NPZ files (each 400-580 MB on disk,
# expands to ~5-6 GB float64 for the peak-layer slice). Loads one model at a
# time, but the permutation test requires reloading, so peak memory is ~12 GB
# per model. Well within 256 GB.
#
# Output: results/clinical_dissociation/subspace_dissociation.json
#         figures/subspace_dissociation.png

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "===== subspace_dissociation ====="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Mem: $SLURM_MEM_PER_NODE MB"
echo ""

python -u src/subspace_dissociation.py

echo ""
echo "===== DONE: $(date) ====="
