#!/bin/bash
# Sync experiment code + prepared data to HPC3
# Run from HPC2: bash experiments/scripts/sync_to_hpc3.sh

SSH_KEY="/hpc2hdd/home/mzhang630/data/id_rsa"
HPC3="mzhang630@hpc3login.hpc.hkust-gz.edu.cn"
LOCAL_BASE="/hpc2hdd/home/mzhang630/data/nature/experiments"
REMOTE_BASE="/data/user/mzhang630/data/nature_exp"

echo "=== Syncing to HPC3 ==="

# Create remote directories
echo "[1/4] Creating remote directories..."
ssh -i ${SSH_KEY} ${HPC3} "mkdir -p ${REMOTE_BASE}/{stimuli,activations,modules,results/dissociation,results/pruning,results/scaling}"

# Sync stimuli data
echo "[2/4] Syncing stimuli..."
scp -i ${SSH_KEY} ${LOCAL_BASE}/data/stimuli_full.jsonl ${HPC3}:${REMOTE_BASE}/stimuli/
scp -i ${SSH_KEY} ${LOCAL_BASE}/data/stimuli_pilot.jsonl ${HPC3}:${REMOTE_BASE}/stimuli/

# Sync experiment code
echo "[3/4] Syncing experiment code..."
scp -i ${SSH_KEY} -r ${LOCAL_BASE}/src/ ${HPC3}:${REMOTE_BASE}/src/
scp -i ${SSH_KEY} -r ${LOCAL_BASE}/configs/ ${HPC3}:${REMOTE_BASE}/configs/
scp -i ${SSH_KEY} -r ${LOCAL_BASE}/scripts/ ${HPC3}:${REMOTE_BASE}/scripts/
scp -i ${SSH_KEY} -r ${LOCAL_BASE}/setup/ ${HPC3}:${REMOTE_BASE}/setup/

# Sync Pythia-410M if it exists
PYTHIA_DIR="/hpc2hdd/home/mzhang630/data/nature/models/pythia-410m"
if [ -f "${PYTHIA_DIR}/model.safetensors" ]; then
    FSIZE=$(stat -c %s "${PYTHIA_DIR}/model.safetensors" 2>/dev/null || echo 0)
    if [ "$FSIZE" -gt 500000000 ]; then
        echo "[4/4] Syncing Pythia-410M model..."
        ssh -i ${SSH_KEY} ${HPC3} "mkdir -p ${REMOTE_BASE}/models/pythia-410m"
        scp -i ${SSH_KEY} ${PYTHIA_DIR}/* ${HPC3}:${REMOTE_BASE}/models/pythia-410m/
    else
        echo "[4/4] Pythia-410M incomplete (${FSIZE} bytes), skipping. Will use Qwen for pilot."
    fi
else
    echo "[4/4] No Pythia-410M found, will use Qwen for pilot."
fi

# Sync wheel files for offline pip install
WHEELS_DIR="/hpc2hdd/home/mzhang630/data/nature/wheels"
if [ -d "${WHEELS_DIR}" ] && [ "$(ls -A ${WHEELS_DIR})" ]; then
    echo "[5/5] Syncing wheel files for offline install..."
    ssh -i ${SSH_KEY} ${HPC3} "mkdir -p ${REMOTE_BASE}/wheels"
    scp -i ${SSH_KEY} ${WHEELS_DIR}/*.whl ${HPC3}:${REMOTE_BASE}/wheels/ 2>/dev/null
    echo "  On HPC3, install missing packages with:"
    echo "    conda activate alphasteer"
    echo "    pip install --no-index --find-links=${REMOTE_BASE}/wheels/ scikit-learn scipy networkx"
fi

echo ""
echo "=== Sync complete ==="
echo "On HPC3, run:"
echo "  # 1. Install missing packages"
echo "  source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate alphasteer"
echo "  pip install --no-index --find-links=${REMOTE_BASE}/wheels/ scikit-learn scipy networkx"
echo "  # 2. Submit pilot"
echo "  sbatch ${REMOTE_BASE}/scripts/slurm/pilot_quick_test.sh"
