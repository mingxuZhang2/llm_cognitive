#!/bin/bash
# Pull RSA activations from HPC3 and run the full local analysis pipeline.
# Steps:
#   1. scp {model}_rsa_activations.npz files from HPC3
#   2. compute_rsa.py per model -> {model}_rsa.json + {model}_rsa_llm_rdms.npz
#   3. rsa_cross_model_analysis.py -> figures + cross_model_summary.json

set -e
BASE="/hpc2hdd/home/mzhang630/data/nature/experiments"
RES="${BASE}/results/cognitive_rsa"
HPC3_RES="/data/user/mzhang630/data/nature_exp/results/cognitive_rsa"
SSH_KEY="/hpc2hdd/home/mzhang630/data/id_rsa"
SSH_HOST="mzhang630@hpc3login.hpc.hkust-gz.edu.cn"
SCP="scp -i ${SSH_KEY} -o StrictHostKeyChecking=no"

mkdir -p "${RES}"

echo "[1/3] Pulling RSA activation NPZs from HPC3..."
for model in Qwen2.5-7B-Instruct Meta-Llama-3.1-8B-Instruct Mistral-7B-Instruct-v0.3 gemma-2-9b-it; do
    ${SCP} "${SSH_HOST}:${HPC3_RES}/${model}_rsa_activations.npz" "${RES}/" || \
        echo "  WARN: missing ${model} activations"
done

echo
echo "[2/3] Computing per-model RSA (Spearman vs brain RDM, 10K permutations)..."
for model in Qwen2.5-7B-Instruct Meta-Llama-3.1-8B-Instruct Mistral-7B-Instruct-v0.3 gemma-2-9b-it; do
    if [ -f "${RES}/${model}_rsa_activations.npz" ]; then
        echo "  --- ${model} ---"
        python "${BASE}/src/compute_rsa.py" \
            --brain_rdm_npz "${RES}/brain_rdm.npz" \
            --activations_npz "${RES}/${model}_rsa_activations.npz" \
            --output_path "${RES}/${model}_rsa.json" \
            --n_permutations 10000
    fi
done

echo
echo "[3/3] Cross-model synthesis + figures..."
python "${BASE}/src/rsa_cross_model_analysis.py"

echo
echo "DONE. Outputs:"
ls -la "${RES}/"*.json "${BASE}/figures/cognitive_rsa_"*.png 2>/dev/null
