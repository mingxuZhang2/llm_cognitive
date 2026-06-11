#!/bin/bash
#SBATCH --job-name=cf_cross
#SBATCH --partition=i64m1tga800u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/cf_cross_%j.out

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /hpc2hdd/home/mzhang630/data/nature/experiments

# Model paths
QWEN="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LLAMA="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ac6a82c495cbcdf8e379063f645bb928ff65be7f"
MISTRAL="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
GEMMA="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"

echo "=== Content x Format Crossed Design — $(date) ==="

# First generate stimuli (CPU, fast)
echo ""
echo "--- Generating stimuli ---"
python3 -u src/content_format_cross_stimuli.py

# Then run extraction + analysis + behavioral test for all 4 models
for MODEL_PATH in "$QWEN" "$LLAMA" "$MISTRAL" "$GEMMA"; do
    SHORT=$(basename $(dirname $(dirname "$MODEL_PATH")) | sed 's/models--[^-]*--//')
    echo ""
    echo "=== $SHORT ==="
    echo "--- RSA extraction + analysis ---"
    python3 -u src/content_format_cross_rsa.py --model_path "$MODEL_PATH" --model_short "$SHORT"
    echo "--- Behavioral format transfer ---"
    python3 -u src/format_transfer_behavior.py --model_path "$MODEL_PATH" --model_short "$SHORT"
done

echo ""
echo "=== ALL DONE — $(date) ==="
