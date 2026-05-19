#!/bin/bash
#SBATCH --job-name=funcatlas_pilot
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/pilot_%j.out
#SBATCH --error=/data/user/mzhang630/logs/pilot_%j.err

# PILOT: Quick test with Pythia-410M + 200 samples
# Validates entire Phase 1-3 pipeline before committing to 7B models
# Submit: sbatch pilot_quick_test.sh

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate funcatlas

SRC="/hpc2hdd/home/mzhang630/data/nature/experiments/src"
BASE="/data/user/mzhang630/data/nature_exp"
MODEL_PATH="/data/user/mzhang630/data/nature_exp/models/pythia-410m"
MODEL_NAME="EleutherAI/pythia-410m"

mkdir -p ${BASE}/{stimuli,activations,modules,results/dissociation} /data/user/mzhang630/logs

echo "============================================"
echo "PILOT TEST: Pythia-410M, 200 samples"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

# Step 1: Prepare pilot stimuli (small set)
echo ""
echo "[Step 1/4] Preparing pilot stimuli..."
cd ${SRC}
python stimulus_preparation.py \
    --output "${BASE}/stimuli/pilot_stimuli.jsonl" \
    --pilot \
    --n_pilot 15

# Step 2: Extract activations
echo ""
echo "[Step 2/4] Extracting activations from Pythia-410M..."
python activation_extraction.py \
    --model "${MODEL_PATH}" \
    --stimuli "${BASE}/stimuli/pilot_stimuli.jsonl" \
    --output_dir "${BASE}/activations" \
    --max_length 256 \
    --batch_size 8

# Step 3: Module discovery
echo ""
echo "[Step 3/4] Running module discovery (K=5,10)..."
MODEL_SHORT="pythia-410m"
python module_discovery.py \
    --activation_path "${BASE}/activations/${MODEL_SHORT}_activations.npy" \
    --output_dir "${BASE}/modules" \
    --K 5 10

# Step 4: Quick dissociation test (module 0 vs module 1)
echo ""
echo "[Step 4/4] Running pilot ablation test..."

# Check which module is which by looking at sample composition
python -c "
import numpy as np
import json

modules = np.load('${BASE}/modules/${MODEL_SHORT}_K10_modules.npz')
neuron_assign = modules['neuron_assign']
sample_assign = modules['sample_assign']

# Load stimuli to check categories
with open('${BASE}/stimuli/pilot_stimuli.jsonl') as f:
    stimuli = [json.loads(line) for line in f]

print('Module composition (K=10):')
for k in range(10):
    s_mask = sample_assign == k
    n_neurons = (neuron_assign == k).sum()
    n_samples = s_mask.sum()
    cats = [stimuli[i]['category'] for i in range(len(stimuli)) if i < len(sample_assign) and sample_assign[i] == k]
    cat_counts = {}
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:3]
    top_str = ', '.join([f'{c}({n})' for c, n in top_cats])
    print(f'  Module {k}: {n_neurons:>6d} neurons, {n_samples:>3d} samples | {top_str}')

# Module sizes summary
total_n = len(neuron_assign)
print(f'\nTotal neurons: {total_n:,}')
print(f'Module size range: {min((neuron_assign==k).sum() for k in range(10)):,} - {max((neuron_assign==k).sum() for k in range(10)):,}')
"

echo ""
echo "============================================"
echo "PILOT COMPLETE"
echo "============================================"
echo "Check results:"
echo "  Activations: ${BASE}/activations/${MODEL_SHORT}_*"
echo "  Modules:     ${BASE}/modules/${MODEL_SHORT}_*"
echo "  Stimuli:     ${BASE}/stimuli/pilot_stimuli.jsonl"
echo ""
echo "If module composition looks reasonable, proceed with:"
echo "  sbatch phase1_extract_activations.sh  (for 7B models)"
