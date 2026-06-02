#!/bin/bash
#SBATCH --job-name=base_1.5b_rsa
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/extract_base_1.5b_%j.out
#SBATCH --error=/hpc2hdd/home/mzhang630/data/nature/experiments/logs/extract_base_1.5b_%j.err

# Extract per-stim RSA activations from the Qwen2.5-1.5B BASE model
# (same procedure as the Instruct scaling sweep, but for the pre-RLHF checkpoint).
# Output: results/cognitive_rsa/Qwen2.5-1.5B_rsa_v2_per_stim.npz
#
# After this completes, re-run:
#   python src/base_vs_instruct_rsa.py

source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate base
export TORCHDYNAMO_DISABLE=1

EXPDIR="/hpc2hdd/home/mzhang630/data/nature/experiments"
MODEL="/hpc2hdd/home/mzhang630/data/models_dl/Qwen2.5-1.5B"
SHORT="Qwen2.5-1.5B"
STIM="${EXPDIR}/data/cognitive_stimuli/rsa/rsa_stimuli.jsonl"
OUTDIR="${EXPDIR}/results/cognitive_rsa"

mkdir -p "${OUTDIR}" "${EXPDIR}/logs"

cd "${EXPDIR}"
echo "Extracting RSA activations: ${SHORT}"
echo "Model: ${MODEL}"
echo "Stimuli: ${STIM}"
echo "Output: ${OUTDIR}"

python -u src/extract_rsa_activations_v2.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --stimuli_jsonl "${STIM}" \
    --output_dir "${OUTDIR}"

echo "DONE: ${SHORT}"
echo "Now run: python src/base_vs_instruct_rsa.py"
