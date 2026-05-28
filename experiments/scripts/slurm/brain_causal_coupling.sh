#!/bin/bash
#SBATCH --job-name=brain_coupling
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/mzhang630/logs/brain_coupling_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/brain_coupling_%A_%a.err
#SBATCH --array=0-3

source ~/.bashrc
conda activate alphasteer

BASEDIR=/data/user/mzhang630/data/nature_exp
STIMULI=$BASEDIR/stimuli/rsa_stimuli.jsonl
BRAIN_RDM=$BASEDIR/results/cognitive_rsa/brain_rdm.npz
OUTDIR=$BASEDIR/results/brain_causal_coupling
CODEDIR=/hpc2hdd/home/mzhang630/data/nature/experiments

mkdir -p $OUTDIR /data/user/mzhang630/logs

MODELS=(
    "/data/user/mzhang630/snapshots/Qwen2.5-7B-Instruct"
    "/data/user/mzhang630/snapshots/Meta-Llama-3.1-8B-Instruct"
    "/data/user/mzhang630/snapshots/Mistral-7B-Instruct-v0.3"
    "/data/user/mzhang630/snapshots/gemma-2-9b-it"
)
SHORTS=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "gemma-2-9b-it"
)

MODEL_PATH=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL_SHORT=${SHORTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Model: $MODEL_SHORT ==="
echo "=== Path: $MODEL_PATH ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

cd $CODEDIR/..

python -m experiments.src.brain_causal_coupling \
    --model_path $MODEL_PATH \
    --model_short $MODEL_SHORT \
    --stimuli $STIMULI \
    --output_dir $OUTDIR \
    --brain_rdm $BRAIN_RDM \
    --n_ablate 5000

echo "=== Done: $MODEL_SHORT ==="
