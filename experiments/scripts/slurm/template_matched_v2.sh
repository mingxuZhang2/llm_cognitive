#!/bin/bash
#SBATCH --job-name=tm_v2
#SBATCH --partition=i64m1tga800u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/tm_v2_%j.out

# Template-matched RSA V2: rich multi-sentence templates.
#
# V2 fixes V1's critical flaw: V1 used single-sentence templates (20-28 words)
# that destroyed social cognition content (false belief, ToM, intention).
# V2 uses 3-4 sentence templates (~55-70 words) that preserve the cognitive
# structure of all 14 conditions while controlling format.
#
# Step 1: Generate 840 V2 stimuli (CPU, fast)
# Step 2: Run extraction + analysis for all 4 models (GPU, sequential)

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

EXPDIR=/hpc2hdd/home/mzhang630/data/nature/experiments
cd "$EXPDIR"

mkdir -p logs results/template_matched_rsa_v2

QWEN="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LLAMA="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ac6a82c495cbcdf8e379063f645bb928ff65be7f"
MISTRAL="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
GEMMA="/hpc2hdd/home/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"

echo "=== Template-Matched RSA V2 ($(date)) ==="
echo ""

# ── Step 1: Generate stimuli ──
echo "--- Step 1: Generating V2 stimuli ---"
python3 -u src/template_matched_stimuli_v2.py
echo ""

# ── Step 2: Run extraction + analysis for each model ──
declare -A MODEL_PATHS
MODEL_PATHS["Qwen2.5-7B-Instruct"]="$QWEN"
MODEL_PATHS["Meta-Llama-3.1-8B-Instruct"]="$LLAMA"
MODEL_PATHS["Mistral-7B-Instruct-v0.3"]="$MISTRAL"
MODEL_PATHS["gemma-2-9b-it"]="$GEMMA"

for MODEL_SHORT in "Qwen2.5-7B-Instruct" "Meta-Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.3" "gemma-2-9b-it"; do
    MODEL_PATH="${MODEL_PATHS[$MODEL_SHORT]}"
    echo ""
    echo "=== $MODEL_SHORT ($(date)) ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    python3 -u src/template_matched_rsa_v2.py \
        --model_path "$MODEL_PATH" \
        --model_short "$MODEL_SHORT" \
        --batch_size 8
    echo "=== Done: $MODEL_SHORT ($(date)) ==="
done

echo ""
echo "=== ALL 4 MODELS DONE ($(date)) ==="
echo ""
echo "Results saved to: results/template_matched_rsa_v2/"
ls -la results/template_matched_rsa_v2/
