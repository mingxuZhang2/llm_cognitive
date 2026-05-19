#!/bin/bash
#SBATCH --job-name=funcatlas_full
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/mzhang630/logs/full_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/full_%A_%a.err
#SBATCH --array=0-3

# Full experiment: Phase 1 (extract) + Phase 2 (discover) for one model
# Runs on the FULL stimulus set (3133 samples)
# 4 models in parallel via array jobs → 4 GPUs
# Submit: sbatch full_experiment.sh

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

# Model configs: HF_name|short_name|snapshot_path
CONFIGS=(
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    "camenduru/Meta-Llama-3.1-8B-Instruct|Meta-Llama-3.1-8B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/$(ls /data/user/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ 2>/dev/null | head -1)"
    "mistralai/Mistral-7B-Instruct-v0.3|Mistral-7B-Instruct-v0.3|/data/user/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/$(ls /data/user/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/ 2>/dev/null | head -1)"
    "google/gemma-2-9b-it|gemma-2-9b-it|/data/user/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/$(ls /data/user/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/ 2>/dev/null | head -1)"
)

IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_HF="${CFG[0]}"
MODEL_SHORT="${CFG[1]}"
MODEL_PATH="${CFG[2]}"

mkdir -p ${BASE}/{activations,modules,results} /data/user/mzhang630/logs

echo "============================================"
echo "Full Experiment: ${MODEL_SHORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Stimuli: full set (3133 samples)"
echo "============================================"

cd ${SRC}

# ==========================================
# Phase 1: Activation Extraction
# ==========================================
ACT_FILE="${BASE}/activations/${MODEL_SHORT}_activations.npy"
if [ -f "${ACT_FILE}" ]; then
    echo "[Phase 1] Activations already exist, skipping."
else
    echo "[Phase 1] Extracting activations..."
    python activation_extraction.py \
        --model "${MODEL_PATH}" \
        --model_short "${MODEL_SHORT}" \
        --stimuli "${BASE}/stimuli/stimuli_full.jsonl" \
        --output_dir "${BASE}/activations" \
        --max_length 512 \
        --batch_size 32
fi

# ==========================================
# Phase 2: Module Discovery (GPU-accelerated)
# ==========================================
echo ""
echo "[Phase 2] Running GPU module discovery (K=5,8,10,15,20)..."
python module_discovery.py \
    --activation_path "${BASE}/activations/${MODEL_SHORT}_activations.npy" \
    --output_dir "${BASE}/modules" \
    --K 5 8 10 15 20

# ==========================================
# Quick analysis: module composition
# ==========================================
echo ""
echo "[Analysis] Module composition at K=8:"
python -c "
import numpy as np, json
na = np.load('${BASE}/modules/${MODEL_SHORT}_K8_modules.npz')['neuron_assign']
sa = np.load('${BASE}/modules/${MODEL_SHORT}_K8_modules.npz')['sample_assign']
with open('${BASE}/stimuli/stimuli_full.jsonl') as f:
    stimuli = [json.loads(l) for l in f]
for k in range(8):
    si = np.where(sa == k)[0]
    cats = {}
    for i in si:
        if i < len(stimuli):
            cats[stimuli[i]['category']] = cats.get(stimuli[i]['category'], 0) + 1
    top = sorted(cats.items(), key=lambda x: -x[1])[:3]
    ts = ', '.join([f'{c}({n})' for c, n in top])
    nn = int((na == k).sum())
    pur = max(cats.values()) / sum(cats.values()) if cats else 0
    print(f'  Mod {k}: {nn:>7,} neurons, {len(si):>4} samples, pur={pur:.2f} | {ts}')
"

echo ""
echo "============================================"
echo "DONE: ${MODEL_SHORT}"
echo "Phase 1+2 complete. Ready for Phase 3 (dissociation)."
echo "============================================"
