#!/bin/bash
#SBATCH --job-name=beh_vlm
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/mzhang630/logs/beh_vlm_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/beh_vlm_%A_%a.err
#SBATCH --array=0-1

# Array 0: Behavioral prediction (Qwen2.5-3B-Instruct)
# Array 1: Behavioral prediction (Qwen2.5-1.5B-Instruct) for cross-model validation

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
STIM="${BASE}/cognitive_stimuli/rsa/rsa_stimuli.jsonl"
BRAIN="${BASE}/results/cognitive_rsa/brain_rdm.npz"
OUTDIR="${BASE}/results/behavioral_prediction"
mkdir -p "${OUTDIR}"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    SHORT="Qwen2.5-3B-Instruct"
else
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
    SHORT="Qwen2.5-1.5B-Instruct"
fi

cd "${BASE}/src"
python predict_behavioral_failure.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --stimuli_jsonl "${STIM}" \
    --brain_rdm "${BRAIN}" \
    --output_dir "${OUTDIR}" \
    --n_trials 20

echo "DONE: ${SHORT}"
