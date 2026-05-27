#!/bin/bash
#SBATCH --job-name=spec_v3_5k
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:30:00
#SBATCH --output=/data/user/mzhang630/logs/spec_v3_5k_%A_%a.out
#SBATCH --error=/data/user/mzhang630/logs/spec_v3_5k_%A_%a.err
#SBATCH --array=0-3

# Critical test: at n=5000, does the v3 CONTRAST atlas preserve general
# language modeling? If yes, the v3 localization claim is salvaged.
# If no, the v3 atlas is also load-bearing-language confounded.
#
# This is a parallel of specificity_control.py but uses contrast_attribution
# (v3 atlas) instead of v1 moral_selectivity, and n_ablate=5000 instead of 500.

set -e
source /data/user/mzhang630/miniconda3/etc/profile.d/conda.sh
conda activate alphasteer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

SRC="/data/user/mzhang630/data/nature_exp/src"
BASE="/data/user/mzhang630/data/nature_exp"
STIM_DIR="${BASE}/cognitive_stimuli"

CONFIGS=(
    "Qwen2.5-7B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    "Meta-Llama-3.1-8B-Instruct|/data/user/mzhang630/.cache/huggingface/hub/models--camenduru--Meta-Llama-3.1-8B-Instruct/snapshots/ac6a82c495cbcdf8e379063f645bb928ff65be7f"
    "Mistral-7B-Instruct-v0.3|/data/user/mzhang630/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    "gemma-2-9b-it|/data/user/mzhang630/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
)
IFS='|' read -ra CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SHORT="${CFG[0]}"
MODEL_PATH="${CFG[1]}"

OUTDIR="${BASE}/results/specificity_v3_n5000"
META_FILE="${BASE}/activations/${MODEL_SHORT}_cognitive_pilot_meta.json"
ATTR_V3="${BASE}/results/contrast_pilot/${MODEL_SHORT}_contrast_attribution.npz"
mkdir -p "${OUTDIR}" /data/user/mzhang630/logs

cd "${SRC}"
# Wrap specificity_control.py via a small inline patch to use n=5000.
python -c "
import sys
sys.argv = ['', '--model_path', '${MODEL_PATH}',
            '--model_short', '${MODEL_SHORT}',
            '--attribution_npz', '${ATTR_V3}',
            '--decomposition_jsonl', '${STIM_DIR}/decomposition_stimuli.jsonl',
            '--meta_path', '${META_FILE}',
            '--output_dir', '${OUTDIR}']
import specificity_control
# Monkey-patch n_ablate from 500 to 5000
_orig_run = specificity_control.run
def patched_run(*args, **kwargs):
    # The select_neurons call uses n_ablate=500 hardcoded; need to change it.
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from moral_decomposition import (
        install_ablation_hooks, compute_wrongness_scores, pair_discrimination,
        resolve_rating_token_ids, select_neurons,
    )
    from specificity_control import NEUTRAL_TEXTS, VALENCE_PAIRS, compute_ppl_on_texts, rate_neutral, valence_discrimination
    import time, json
    from pathlib import Path

    model_path, model_short, attribution_npz, decomposition_jsonl, meta_path, output_dir = args
    device = kwargs.get('device', 'cuda')
    N_ABLATE = 5000

    t0 = time.time()
    print(f'='*70)
    print(f'v3 Specificity (n={N_ABLATE}): {model_short}')
    print(f'='*70)
    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta['n_layers']; ffn_dim = meta['ffn_dim']; n_neurons = n_layers * ffn_dim
    with open(decomposition_jsonl) as f:
        stimuli = [json.loads(line) for line in f if line.strip()]
    norm_stim = [s for s in stimuli if s['condition'] == 'norm_type']

    print(f'  Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rating_token_ids = resolve_rating_token_ids(tokenizer, norm_stim[0]['text'])

    def run_all_measurements():
        m = {}
        ns, _, _ = compute_wrongness_scores(model, tokenizer, norm_stim, rating_token_ids, device)
        m['norm_type'] = pair_discrimination(norm_stim, ns)[0].get('norm_type', {})
        m['neutral_rating'] = rate_neutral(model, tokenizer, rating_token_ids, device)
        m['valence'] = valence_discrimination(model, tokenizer, rating_token_ids, device)
        ppls = compute_ppl_on_texts(model, tokenizer, NEUTRAL_TEXTS, device)
        m['neutral_ppl'] = {'mean': float(np.mean(ppls)), 'median': float(np.median(ppls)),
                            'max': float(np.max(ppls)), 'min': float(np.min(ppls))}
        return m

    print(f'Baseline...')
    baseline = run_all_measurements()
    print(f'  norm_type: {baseline[\"norm_type\"][\"mean\"]:+.2f}')
    print(f'  neutral_rating: {baseline[\"neutral_rating\"][\"mean_rating\"]:.2f}')
    print(f'  valence: {baseline[\"valence\"][\"mean\"]:+.2f}')
    print(f'  neutral_ppl: {baseline[\"neutral_ppl\"][\"mean\"]:.1f}')

    print(f'Ablate top-{N_ABLATE} contrast (v3 atlas)...')
    top, _ = select_neurons(attribution_npz, 'norm_type', N_ABLATE, side='top')
    hooks = install_ablation_hooks(model, top, n_layers, ffn_dim, device)
    ablated = run_all_measurements()
    for h in hooks:
        h.remove()
    print(f'  norm_type: {ablated[\"norm_type\"][\"mean\"]:+.2f}  (Δ {ablated[\"norm_type\"][\"mean\"]-baseline[\"norm_type\"][\"mean\"]:+.2f})')
    print(f'  neutral_rating: {ablated[\"neutral_rating\"][\"mean_rating\"]:.2f}  (Δ {ablated[\"neutral_rating\"][\"mean_rating\"]-baseline[\"neutral_rating\"][\"mean_rating\"]:+.2f})')
    print(f'  valence: {ablated[\"valence\"][\"mean\"]:+.2f}  (Δ {ablated[\"valence\"][\"mean\"]-baseline[\"valence\"][\"mean\"]:+.2f})')
    print(f'  neutral_ppl: {ablated[\"neutral_ppl\"][\"mean\"]:.1f}  ({ablated[\"neutral_ppl\"][\"mean\"]/baseline[\"neutral_ppl\"][\"mean\"]:.2f}x)')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = {'model': model_short, 'n_ablate': N_ABLATE, 'baseline': baseline,
                'ablated': ablated, 'elapsed_seconds': time.time() - t0}
    with open(Path(output_dir) / f'{model_short}_specificity_v3_n5000.json', 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Done. elapsed {time.time()-t0:.0f}s')

patched_run('${MODEL_PATH}', '${MODEL_SHORT}', '${ATTR_V3}',
            '${STIM_DIR}/decomposition_stimuli.jsonl',
            '${META_FILE}', '${OUTDIR}')
"
echo "DONE: ${MODEL_SHORT}"
