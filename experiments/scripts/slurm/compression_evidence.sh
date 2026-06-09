#!/bin/bash
#SBATCH --job-name=compress
#SBATCH --partition=i64m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/compress_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "=== Emotion Compression Evidence ($(date)) ==="
python3 -u src/emotion_compression_evidence.py
echo "=== DONE ($(date)) ==="
