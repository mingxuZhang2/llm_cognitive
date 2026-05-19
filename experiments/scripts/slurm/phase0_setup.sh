#!/bin/bash
#SBATCH --job-name=funcatlas_setup
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/setup_%j.out
#SBATCH --error=/data/user/mzhang630/logs/setup_%j.err

# Phase 0: Environment setup + stimulus preparation + model download
# Submit: sbatch phase0_setup.sh

mkdir -p /data/user/mzhang630/logs

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh

# Create environment
bash /hpc2hdd/home/mzhang630/data/nature/experiments/setup/create_env.sh
conda activate alphasteer

# Prepare stimuli
echo "Preparing stimulus dataset..."
python /hpc2hdd/home/mzhang630/data/nature/experiments/src/stimulus_preparation.py \
    --output /data/user/mzhang630/data/nature_exp/stimuli/all_stimuli.jsonl \
    --n_per_category 400

# Pre-download models
echo "Pre-downloading models..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['HF_HOME'] = '/data/user/mzhang630/data/nature_exp/models'

models = [
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'google/gemma-2-9b-it',
]
for m in models:
    print(f'Downloading {m}...')
    AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto', trust_remote_code=True)
    print(f'  Done: {m}')
"

echo "Setup complete."
