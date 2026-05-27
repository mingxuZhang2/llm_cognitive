#!/bin/bash
#SBATCH --job-name=vlm_cmp2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/mzhang630/logs/vlm_cmp2_%j.out
#SBATCH --error=/data/user/mzhang630/logs/vlm_cmp2_%j.err

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export TORCHDYNAMO_DISABLE=1

BASE="/data/user/mzhang630/data/nature_exp"

cd "${BASE}/src"
python -c "
import json, torch, numpy as np
from transformers import AutoTokenizer, AutoConfig

model_path = '/data/user/mzhang630/data/models_dl/Qwen2.5-VL-3B-Instruct'
model_short = 'Qwen2.5-VL-3B-Instruct'

# Load stimuli
stimuli = [json.loads(l) for l in open('${BASE}/cognitive_stimuli/rsa/rsa_stimuli.jsonl')]
texts = [s.get('text', s.get('prompt', '')) for s in stimuli]
conditions = [s.get('condition', s.get('category')) for s in stimuli]
print(f'{len(stimuli)} stimuli')

# Load VL model bypassing AutoProcessor (no torchvision needed)
print(f'Loading {model_path}...')
from transformers import Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
    trust_remote_code=True,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Extract hidden states from the language model backbone
all_acts = []
for i in range(0, len(texts), 8):
    batch = texts[i:i+8]
    inputs = tokenizer(batch, return_tensors='pt', padding=True,
                      truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Access the language model backbone (model.model.language_model)
    lm = model.model.language_model
    with torch.no_grad():
        out = lm(input_ids=inputs['input_ids'],
                 attention_mask=inputs['attention_mask'],
                 output_hidden_states=True)
        hidden_states_list = out.hidden_states  # tuple of (batch, seq, hidden)

    for b in range(len(batch)):
        mask = inputs['attention_mask'][b].bool()
        layers = []
        for hs in hidden_states_list:
            tokens = hs[b][mask]
            mean_pooled = tokens.mean(dim=0).cpu().numpy()
            layers.append(mean_pooled)
        all_acts.append(np.stack(layers))

    if i % 100 == 0:
        print(f'  {i}/{len(texts)}')

all_acts = np.stack(all_acts)  # (n_stim, n_layers, hidden)
print(f'Activations: {all_acts.shape}')

# Save in same format as extract_rsa_activations_v2
# Reshape to match expected format: (n_pooling, n_stim, n_layers, hidden)
# We only have mean_all pooling
acts_expanded = all_acts[np.newaxis, ...]  # (1, n_stim, n_layers, hidden)
layer_names = [f'layer_{i}' for i in range(all_acts.shape[1])]

np.savez_compressed(
    f'${BASE}/results/cognitive_rsa/{model_short}_rsa_v2_per_stim.npz',
    per_stim_activations=acts_expanded.astype(np.float16),
    conditions=np.array(conditions),
    pooling_names=np.array(['mean_all']),
    layer_names=np.array(layer_names),
)
print(f'Saved {model_short}_rsa_v2_per_stim.npz')

# Quick RSA check
from scipy.stats import spearmanr
brain = np.load('${BASE}/results/cognitive_rsa/brain_rdm.npz', allow_pickle=True)
brain_rdm = brain['rdm']
brain_conds = list(brain['conditions'])
unique_conds = sorted(set(conditions))
stim_cond = np.array([unique_conds.index(c) for c in conditions])
order = [unique_conds.index(c) for c in brain_conds]
triu = np.triu_indices(len(brain_conds), k=1)
brain_vec = brain_rdm[triu]

best_rho, best_L = -1, 0
for L in range(all_acts.shape[1]):
    cond_means = np.zeros((len(unique_conds), all_acts.shape[2]))
    for ci in range(len(unique_conds)):
        idx = np.where(stim_cond == ci)[0]
        cond_means[ci] = all_acts[idx, L, :].mean(axis=0)
    cond_means = cond_means[order]
    cm = cond_means - cond_means.mean(axis=0)
    norms = np.linalg.norm(cm, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    rdm = 1.0 - np.clip((cm/norms) @ (cm/norms).T, -1, 1)
    rho, _ = spearmanr(brain_vec, rdm[triu])
    if rho > best_rho:
        best_rho, best_L = rho, L

print(f'\n=== VLM Brain-LLM RSA ===')
print(f'  Peak L{best_L}: rho = {best_rho:+.4f}')
print(f'  (Compare with text-only Qwen2.5-3B: rho ~ 0.66)')
"

echo "DONE"
