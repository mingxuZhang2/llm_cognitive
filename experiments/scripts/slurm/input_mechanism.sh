#!/bin/bash
#SBATCH --job-name=input_mech
#SBATCH --partition=i64m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/input_mech_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /hpc2hdd/home/mzhang630/data/nature/experiments

echo "=== Emotion Input Mechanism Analysis ($(date)) ==="
python3 -u src/emotion_input_mechanism.py
echo "=== DONE ($(date)) ==="
