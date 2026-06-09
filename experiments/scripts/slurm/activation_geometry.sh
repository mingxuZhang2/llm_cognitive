#!/bin/bash
#SBATCH --job-name=act_geom
#SBATCH --partition=i64m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/act_geom_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "=== Activation geometry analysis ($(date)) ==="
python3 src/activation_geometry.py
echo "=== DONE ($(date)) ==="
