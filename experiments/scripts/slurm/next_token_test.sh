#!/bin/bash
#SBATCH --job-name=nxt_tok
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/nxt_tok_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/nxt_tok_%A_%a.err
#SBATCH --array=0-1

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"
STIM="${BASE}/cognitive_stimuli/rsa/rsa_stimuli.jsonl"
BRAIN="${BASE}/results/cognitive_rsa/brain_rdm.npz"
OUTDIR="${BASE}/results/next_token"
mkdir -p "${OUTDIR}"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-3B-Instruct"
    SHORT="Qwen2.5-3B-Instruct"
    LLM_RDM="${BASE}/results/cognitive_rsa/Qwen2.5-3B-Instruct_rsa_v2_per_stim.npz"
else
    MODEL="/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct"
    SHORT="Qwen2.5-1.5B-Instruct"
    LLM_RDM="${BASE}/results/cognitive_rsa/Qwen2.5-1.5B-Instruct_rsa_v2_per_stim.npz"
fi

cd "${BASE}/src"
python next_token_similarity.py \
    --model_path "${MODEL}" \
    --model_short "${SHORT}" \
    --stimuli_jsonl "${STIM}" \
    --brain_rdm "${BRAIN}" \
    --llm_rdm "${LLM_RDM}" \
    --output_dir "${OUTDIR}"

echo "DONE: ${SHORT}"
