#!/bin/bash
# Master orchestration: submit all phases in dependency order
# Usage: bash run_all.sh
# Uses 16 GPUs (2 nodes) with dependency chains

SLURM_DIR="/hpc2hdd/home/mzhang630/data/nature/experiments/scripts/slurm"
mkdir -p /data/user/mzhang630/logs

echo "=========================================="
echo "Functional Atlas Pipeline - 16 GPU Config"
echo "=========================================="

# Phase 0: Setup (env + stimuli + model download)
echo "[Phase 0] Submitting setup..."
P0=$(sbatch --parsable ${SLURM_DIR}/phase0_setup.sh)
echo "  Job ID: ${P0}"

# Phase 1: Activation extraction (4 models, 1 GPU each → 4 GPUs)
echo "[Phase 1] Submitting activation extraction (depends on Phase 0)..."
P1=$(sbatch --parsable --dependency=afterok:${P0} ${SLURM_DIR}/phase1_extract_activations.sh)
echo "  Job IDs: ${P1} (array 0-3)"

# Phase 2: Module discovery (4 models, CPU-only)
echo "[Phase 2] Submitting module discovery (depends on Phase 1)..."
P2=$(sbatch --parsable --dependency=afterok:${P1} ${SLURM_DIR}/phase2_module_discovery.sh)
echo "  Job IDs: ${P2} (array 0-3)"

# Phase 5: Scaling law (independent, can start after Phase 0)
echo "[Phase 5] Submitting scaling law (depends on Phase 0, runs parallel)..."
P5=$(sbatch --parsable --dependency=afterok:${P0} ${SLURM_DIR}/phase5_scaling.sh)
echo "  Job ID: ${P5}"

# Phase 3: Double dissociation (depends on Phase 2)
echo "[Phase 3] Submitting double dissociation (depends on Phase 2)..."
P3=$(sbatch --parsable --dependency=afterok:${P2} ${SLURM_DIR}/phase3_double_dissociation.sh)
echo "  Job IDs: ${P3} (array 0-23)"

# Phase 4: Pruning (depends on Phase 2)
echo "[Phase 4] Submitting pruning experiments (depends on Phase 2)..."
P4=$(sbatch --parsable --dependency=afterok:${P2} ${SLURM_DIR}/phase4_pruning.sh)
echo "  Job IDs: ${P4} (array 0-5)"

# Phase 6: Brain comparison (depends on Phase 2, CPU-only)
echo "[Phase 6] Submitting brain comparison (depends on Phase 2)..."
P6=$(sbatch --parsable --dependency=afterok:${P2} ${SLURM_DIR}/phase6_brain.sh)
echo "  Job ID: ${P6}"

echo ""
echo "=========================================="
echo "Pipeline submitted. Dependency graph:"
echo "=========================================="
echo ""
echo "  Phase 0 (setup)          → ${P0}"
echo "    ├── Phase 1 (extract)  → ${P1}"
echo "    │     └── Phase 2 (discover) → ${P2}"
echo "    │           ├── Phase 3 (dissociation) → ${P3}"
echo "    │           ├── Phase 4 (pruning)      → ${P4}"
echo "    │           └── Phase 6 (brain)        → ${P6}"
echo "    └── Phase 5 (scaling)  → ${P5}"
echo ""
echo "GPU usage timeline:"
echo "  Phase 0: 1 GPU,  ~4h"
echo "  Phase 1: 4 GPUs, ~6h  (parallel with Phase 5: 1 GPU)"
echo "  Phase 2: 0 GPUs, ~8h  (CPU only)"
echo "  Phase 3: up to 16 GPUs, ~6h"
echo "  Phase 4: 6 GPUs, ~12h"
echo "  Phase 5: 1 GPU,  ~24h (running independently)"
echo "  Phase 6: 0 GPUs, ~2h"
echo ""
echo "Estimated wall time: ~3-4 days (with 16 GPU budget)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    /data/user/mzhang630/logs/"
