#!/bin/bash
#SBATCH --job-name=tm_compress
#SBATCH --partition=i64m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/tm_compress_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "=== Template-Matched Compression ($(date)) ==="
python3 -u src/template_matched_compression.py
echo "=== DONE ($(date)) ==="
