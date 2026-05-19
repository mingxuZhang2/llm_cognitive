#!/bin/bash
#SBATCH --job-name=funcatlas_pilot
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/pilot_%j.out
#SBATCH --error=/data/user/mzhang630/logs/pilot_%j.err

# PILOT: Quick test with Qwen2.5-7B (already on HPC3) + pilot stimuli
# Validates Phase 1-3 pipeline
# Submit: sbatch pilot_quick_test.sh

set -e

source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"

# Use Qwen from HPC3 cache (already downloaded)
export HF_HOME="/data/user/mzhang630/.cache/huggingface"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT="Qwen2.5-7B-Instruct"

mkdir -p ${BASE}/{activations,modules,results/dissociation} /data/user/mzhang630/logs

echo "============================================"
echo "PILOT TEST: ${MODEL_NAME}"
echo "Stimuli: pilot set (~120 samples)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

# Step 1: Extract activations
echo ""
echo "[Step 1/3] Extracting activations..."
cd ${SRC}
python activation_extraction.py \
    --model "${MODEL_NAME}" \
    --stimuli "${BASE}/stimuli/stimuli_pilot.jsonl" \
    --output_dir "${BASE}/activations" \
    --max_length 256 \
    --batch_size 32

# Step 2: Module discovery (K=5 and K=8 for pilot)
echo ""
echo "[Step 2/3] Running module discovery..."
python module_discovery.py \
    --activation_path "${BASE}/activations/${MODEL_SHORT}_activations.npy" \
    --output_dir "${BASE}/modules" \
    --K 5 8

# Step 3: Analyze module composition
echo ""
echo "[Step 3/3] Analyzing module composition..."
python -c "
import numpy as np, json

for K in [5, 8]:
    print(f'\n=== K={K} ===')
    modules = np.load('${BASE}/modules/${MODEL_SHORT}_K{K}_modules.npz'.replace('{K}', str(K)))
    neuron_assign = modules['neuron_assign']
    sample_assign = modules['sample_assign']

    with open('${BASE}/stimuli/stimuli_pilot.jsonl') as f:
        stimuli = [json.loads(line) for line in f]

    for k in range(K):
        n_neurons = (neuron_assign == k).sum()
        s_indices = np.where(sample_assign == k)[0]
        cats = {}
        for i in s_indices:
            if i < len(stimuli):
                c = stimuli[i]['category']
                cats[c] = cats.get(c, 0) + 1
        top = sorted(cats.items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join([f'{c}({n})' for c,n in top])
        # Purity: fraction of samples from dominant category
        purity = max(cats.values()) / sum(cats.values()) if cats else 0
        print(f'  Module {k}: {n_neurons:>6,} neurons, {len(s_indices):>3} samples, purity={purity:.2f} | {top_str}')

    # Check math vs code separation
    math_modules = set()
    code_modules = set()
    for k in range(K):
        s_indices = np.where(sample_assign == k)[0]
        cats = {}
        for i in s_indices:
            if i < len(stimuli):
                c = stimuli[i]['category']
                cats[c] = cats.get(c, 0) + 1
        if cats:
            dominant = max(cats, key=cats.get)
            if dominant == 'math': math_modules.add(k)
            if dominant == 'code': code_modules.add(k)

    if math_modules and code_modules and math_modules.isdisjoint(code_modules):
        print(f'  >>> Math/Code SEPARATED: math={math_modules}, code={code_modules}')
    elif math_modules or code_modules:
        print(f'  >>> Partial separation: math={math_modules}, code={code_modules}')
    else:
        print(f'  >>> Math/Code NOT clearly separated at K={K}')
"

echo ""
echo "============================================"
echo "PILOT COMPLETE"
echo "============================================"
echo "Check results in: ${BASE}/activations/ and ${BASE}/modules/"
echo ""
echo "KEY QUESTION: Are math and code samples in different modules?"
echo "If yes → proceed to full experiment with double dissociation"
echo "If no  → try larger K or switch to SAE-based module discovery"
