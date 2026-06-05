#!/bin/bash
#SBATCH --job-name=mech_layer
#SBATCH --partition=i64m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mech_layer_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "=== Layer boundary emergence ($(date)) ==="
python3 src/layer_boundary_emergence.py
echo "=== DONE ($(date)) ==="
