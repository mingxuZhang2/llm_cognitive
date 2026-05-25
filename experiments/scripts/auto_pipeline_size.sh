#!/bin/bash
# Once a Qwen2.5-{size}-Instruct download completes locally, rsync to HPC3
# and submit a single-model RSA v2 extraction SLURM job.
#
# Usage:
#   bash auto_pipeline_size.sh 0.5B
#   bash auto_pipeline_size.sh 1.5B
#   ...
set -e

SIZE="$1"
[ -z "$SIZE" ] && { echo "usage: $0 SIZE"; exit 1; }

LOCAL="/hpc2hdd/home/mzhang630/data/models_dl/Qwen2.5-${SIZE}-Instruct"
REMOTE="/data/user/mzhang630/data/models_dl/Qwen2.5-${SIZE}-Instruct"
SSH="ssh -i /hpc2hdd/home/mzhang630/data/id_rsa -o StrictHostKeyChecking=no mzhang630@hpc3login.hpc.hkust-gz.edu.cn"
RSYNC_KEY="-e 'ssh -i /hpc2hdd/home/mzhang630/data/id_rsa -o StrictHostKeyChecking=no'"

echo "[$(date +%H:%M)] Syncing ${LOCAL} -> HPC3"
rsync -av -e "ssh -i /hpc2hdd/home/mzhang630/data/id_rsa -o StrictHostKeyChecking=no" \
      "${LOCAL}/" "mzhang630@hpc3login.hpc.hkust-gz.edu.cn:${REMOTE}/" \
      2>&1 | tail -5

cat > /tmp/rsa_scale_${SIZE}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=rsa_${SIZE}
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/data/user/mzhang630/logs/rsa_scale_${SIZE}_%j.out
#SBATCH --error=/data/user/mzhang630/logs/rsa_scale_${SIZE}_%j.err

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1

cd /data/user/mzhang630/data/nature_exp/src
python extract_rsa_activations_v2.py \\
    --model_path "${REMOTE}" \\
    --model_short "Qwen2.5-${SIZE}-Instruct" \\
    --stimuli_jsonl "/data/user/mzhang630/data/nature_exp/cognitive_stimuli/rsa/rsa_stimuli.jsonl" \\
    --output_dir "/data/user/mzhang630/data/nature_exp/results/cognitive_rsa"
echo "DONE Qwen2.5-${SIZE}-Instruct"
EOF

scp -i /hpc2hdd/home/mzhang630/data/id_rsa -o StrictHostKeyChecking=no \
    /tmp/rsa_scale_${SIZE}.sh \
    mzhang630@hpc3login.hpc.hkust-gz.edu.cn:/data/user/mzhang630/data/nature_exp/

$SSH "cd /data/user/mzhang630/data/nature_exp && sbatch rsa_scale_${SIZE}.sh"
echo "[$(date +%H:%M)] SLURM submitted for Qwen2.5-${SIZE}"
