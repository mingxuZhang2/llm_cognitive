#!/bin/bash
#SBATCH --job-name=emo_pca
#SBATCH --partition=i64m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/emo_pca_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "=== Emotion per-stimulus PCA ($(date)) ==="
python3 src/emotion_perstim_pca.py
echo "=== DONE ($(date)) ==="
