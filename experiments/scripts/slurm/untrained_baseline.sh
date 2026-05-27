#!/bin/bash
#SBATCH --job-name=untrain
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/data/user/mzhang630/logs/untrain_%j.out
#SBATCH --error=/data/user/mzhang630/logs/untrain_%j.err

# Extract RSA activations from a RANDOMLY INITIALIZED Qwen2.5-1.5B
# Same architecture, same tokenizer, but random weights.
# If brain-LLM RSA ρ ≈ 0 → alignment requires training.
# If ρ > 0.3 → alignment may be architectural artifact.

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"

cd "${BASE}/src"
python -c "
import torch, json, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_path = '/data/user/mzhang630/data/models_dl/Qwen2.5-1.5B-Instruct'
print('Loading config and tokenizer from trained model...')
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Initializing model with RANDOM weights...')
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
model = model.to('cuda')
model.eval()
print(f'Random model on cuda, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params')

# Load stimuli
stim_path = '${BASE}/cognitive_stimuli/rsa/rsa_stimuli.jsonl'
stimuli = [json.loads(l) for l in open(stim_path)]
print(f'Loaded {len(stimuli)} stimuli')

conditions = [s.get('condition', s.get('category')) for s in stimuli]
texts = [s.get('text', s.get('prompt', '')) for s in stimuli]
unique_conds = sorted(set(conditions))

# Extract activations (mean-pool, last layer)
all_acts = []
for i in range(0, len(texts), 8):
    batch = texts[i:i+8]
    inputs = tokenizer(batch, return_tensors='pt', padding=True,
                      truncation=True, max_length=256).to('cuda')
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    for b in range(len(batch)):
        mask = inputs['attention_mask'][b].bool()
        act = out.hidden_states[-1][b][mask].mean(dim=0).cpu().numpy()
        all_acts.append(act)
    if i % 100 == 0:
        print(f'  {i}/{len(texts)}')
all_acts = np.array(all_acts)

# Build condition centroids
cond_idx = {c: [] for c in unique_conds}
for i, c in enumerate(conditions):
    cond_idx[c].append(i)
centroids = np.array([all_acts[cond_idx[c]].mean(axis=0) for c in unique_conds])

# RDM
centroids_c = centroids - centroids.mean(axis=0, keepdims=True)
norms = np.linalg.norm(centroids_c, axis=1, keepdims=True)
norms[norms == 0] = 1.0
rdm = 1.0 - np.clip((centroids_c / norms) @ (centroids_c / norms).T, -1, 1)

# Compare with brain
from scipy.stats import spearmanr
brain = np.load('${BASE}/results/cognitive_rsa/brain_rdm.npz', allow_pickle=True)
brain_rdm = brain['rdm']
brain_conds = list(brain['conditions'])
shared = [c for c in unique_conds if c in brain_conds]
s_idx = [unique_conds.index(c) for c in shared]
b_idx = [brain_conds.index(c) for c in shared]
triu = np.triu_indices(len(shared), k=1)
rho, p = spearmanr(brain_rdm[np.ix_(b_idx, b_idx)][triu], rdm[np.ix_(s_idx, s_idx)][triu])
print(f'\n=== UNTRAINED MODEL BASELINE ===')
print(f'Brain-LLM RSA (random weights): rho = {rho:+.4f}, p = {p:.1e}')
print(f'Trained Qwen 1.5B-Instruct:     rho ≈ +0.64  (for reference)')
if abs(rho) < 0.15:
    print('PASS: Random model shows no significant alignment → trained alignment is real')
elif abs(rho) > 0.3:
    print('WARNING: Random model shows substantial alignment → possible architectural confound')
else:
    print('MARGINAL: Random model shows weak alignment → needs investigation')

np.savez('${BASE}/results/cognitive_rsa/untrained_baseline.npz',
    rdm=rdm, conditions=np.array(unique_conds), rho=rho, p=p)
print('Saved untrained_baseline.npz')
"

echo "DONE"
