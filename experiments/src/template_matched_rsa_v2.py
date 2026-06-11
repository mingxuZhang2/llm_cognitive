#!/usr/bin/env python3
"""
Template-matched RSA V2: self-contained GPU extraction + analysis.

V2 uses RICH multi-sentence templates (3-4 sentences, ~55-70 words) that preserve
the cognitive content of all 14 conditions -- especially social cognition conditions
that require multi-character information asymmetry (false belief), multi-perspective
tracking (theory of mind), and indirect evidence (intention). V1's single-sentence
templates destroyed this content, collapsing within-social RSA from 0.55 to 0.12.

Pipeline (all in one script, one model per run):
  1. Load model + tokenizer
  2. Process all 840 V2 stimuli, extract hidden states at peak layer (mean_all pooling)
  3. Build 14x14 condition RDM (condition-mean, center, cosine distance)
  4. Compare to brain RDM (Spearman rho, permutation p-value)
  5. Per-template sub-RSA (each template independently)
  6. Within-block RSA: within-affective (5 emotions excl valence, 10 pairs),
     within-social (7 conditions excl moral, 21 pairs)
  7. Split-half ceiling (odd vs even items)
  8. Layer sweep (RSA at each layer)
  9. Compare to V1 template-matched results

Usage:
  python src/template_matched_rsa_v2.py --model_path /path/to/model --model_short Qwen2.5-7B-Instruct

Output:
  results/template_matched_rsa_v2/{model_short}_tm_v2_results.json
  results/template_matched_rsa_v2/{model_short}_tm_v2_rdm14.npz
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──
BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "template_matched_rsa_v2"
V1_DIR = BASE / "results" / "template_matched_rsa"
STIMULI_PATH = (BASE / "data" / "cognitive_stimuli" / "rsa"
                / "rsa_stimuli_template_matched_v2.jsonl")

# ── Model configs ──
MODELS = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}

# ── Block definitions ──
AFFECTIVE_CONDITIONS = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
SOCIAL_CONDITIONS = ["belief", "intention", "judgment", "mentalizing", "moral",
                     "empathy", "self_referential", "theory_of_mind"]

# Within-affective: exclude valence (5 conditions, 10 pairs)
WITHIN_AFFECTIVE = ["anger", "fear", "disgust", "sadness", "happiness"]
# Within-social: exclude moral (7 conditions, 21 pairs)
WITHIN_SOCIAL = ["belief", "intention", "mentalizing", "empathy",
                 "self_referential", "theory_of_mind", "judgment"]

# ── RSA recipe (headline) ──
BATCH_SIZE = 8


# ──────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────

def load_stimuli(jsonl_path: str | Path) -> list[dict]:
    """Load stimuli from JSONL file."""
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_centered(act: np.ndarray) -> np.ndarray:
    """Center activations across conditions (subtract condition mean)."""
    return act - act.mean(axis=0, keepdims=True)


def rdm_cosine(act: np.ndarray) -> np.ndarray:
    """Compute cosine distance RDM: 1 - cosine similarity."""
    n = np.linalg.norm(act, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = act / n
    return 1.0 - Xn @ Xn.T


def triu_vals(rdm: np.ndarray) -> np.ndarray:
    """Extract upper triangle values from RDM."""
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def permutation_pvalue(brain_tri: np.ndarray, llm_tri: np.ndarray,
                       n_perm: int = 10000, seed: int = 42) -> float:
    """Permutation null: shuffle condition labels, recompute Spearman."""
    rng = np.random.default_rng(seed)
    n_cond = int((1 + np.sqrt(1 + 8 * len(brain_tri))) / 2)
    obs_rho, _ = spearmanr(brain_tri, llm_tri)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_cond)
        brain_rdm = np.zeros((n_cond, n_cond))
        brain_rdm[np.triu_indices(n_cond, 1)] = brain_tri
        brain_rdm += brain_rdm.T
        perm_rdm = brain_rdm[np.ix_(perm, perm)]
        perm_tri = perm_rdm[np.triu_indices(n_cond, 1)]
        r, _ = spearmanr(perm_tri, llm_tri)
        if r >= obs_rho:
            count += 1
    return count / n_perm


# ──────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, tokenizer, texts: list[str],
                          device: str = "cuda",
                          batch_size: int = BATCH_SIZE,
                          max_length: int = 256) -> np.ndarray:
    """
    Extract mean_all-pooled hidden states for all texts at all layers.

    Returns: np.ndarray [n_texts, n_layers+1, hidden_dim] (float32)
    """
    all_states = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True,
                        add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, use_cache=False)

        # Attention mask for mean pooling
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, seq, 1]

        for layer_states in out.hidden_states:
            # layer_states: [B, seq, hidden]
            pass  # Process below

        # Stack all layers: [n_layers+1, B, seq, hidden]
        stacked = torch.stack([h.float() for h in out.hidden_states], dim=0)
        # Mean over non-pad tokens: [n_layers+1, B, hidden]
        masked = stacked * mask.unsqueeze(0)  # [n_layers+1, B, seq, hidden]
        summed = masked.sum(dim=2)  # [n_layers+1, B, hidden]
        counts = mask.sum(dim=1).unsqueeze(0)  # [1, B, hidden]
        counts = counts.clamp(min=1.0)
        mean_all = (summed / counts).permute(1, 0, 2)  # [B, n_layers+1, hidden]

        all_states.append(mean_all.cpu().numpy().astype(np.float32))

        # Clear GPU memory
        del out, stacked, masked, summed, counts, mean_all, enc
        torch.cuda.empty_cache()

        if (start // batch_size) % 10 == 0:
            print(f"    Processed {min(start + batch_size, len(texts))}/{len(texts)} stimuli")

    return np.concatenate(all_states, axis=0)  # [n_texts, n_layers+1, hidden]


# ──────────────────────────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────────────────────────

def build_condition_rdm(per_stim_states: np.ndarray, conditions: list[str],
                        unique_conds: list[str],
                        layer: int) -> np.ndarray:
    """Build condition-mean RDM at a specific layer."""
    cond_to_idx = {c: [] for c in unique_conds}
    for i, c in enumerate(conditions):
        cond_to_idx[c].append(i)

    act = per_stim_states[:, layer, :].astype(np.float64)
    cmean = np.stack([act[cond_to_idx[c]].mean(axis=0) for c in unique_conds])
    return rdm_cosine(normalize_centered(cmean))


def split_half_ceiling(per_stim_states: np.ndarray, conditions: list[str],
                       unique_conds: list[str], layer: int,
                       n_splits: int = 100, seed: int = 123) -> float:
    """LLM split-half noise ceiling at the headline recipe."""
    rng = np.random.default_rng(seed)
    cond_to_idx = {c: [] for c in unique_conds}
    for i, c in enumerate(conditions):
        cond_to_idx[c].append(i)

    act = per_stim_states[:, layer, :].astype(np.float64)
    rhos = []
    for _ in range(n_splits):
        half_a, half_b = [], []
        for c in unique_conds:
            idx = np.array(cond_to_idx[c])
            rng.shuffle(idx)
            cut = len(idx) // 2
            if cut < 1:
                return float("nan")
            half_a.append(act[idx[:cut]].mean(axis=0))
            half_b.append(act[idx[cut:]].mean(axis=0))

        a = normalize_centered(np.stack(half_a))
        b = normalize_centered(np.stack(half_b))
        rdm_a = rdm_cosine(a)
        rdm_b = rdm_cosine(b)
        r, _ = spearmanr(triu_vals(rdm_a), triu_vals(rdm_b))
        if np.isfinite(r):
            rhos.append(r)

    return float(np.mean(rhos)) if rhos else float("nan")


def per_template_rsa(per_stim_states: np.ndarray, templates: list[str],
                     conditions: list[str], unique_conds: list[str],
                     brain_tri: np.ndarray, layer: int) -> dict:
    """Compute RSA separately for each template."""
    tmpl_set = sorted(set(templates))
    results = {}
    for tmpl in tmpl_set:
        mask = np.array([t == tmpl for t in templates])
        sub_act = per_stim_states[mask, layer, :].astype(np.float64)
        sub_conds = [c for c, m in zip(conditions, mask) if m]

        cond_to_idx = {c: [] for c in unique_conds}
        for i, c in enumerate(sub_conds):
            cond_to_idx[c].append(i)

        cmean = np.stack([sub_act[cond_to_idx[c]].mean(axis=0)
                          for c in unique_conds])
        rdm = rdm_cosine(normalize_centered(cmean))
        r, _ = spearmanr(triu_vals(rdm), brain_tri)
        results[tmpl] = round(float(r), 4)
    return results


def within_block_rsa(rdm: np.ndarray, unique_conds: list[str],
                     brain_rdm: np.ndarray, brain_conds: list[str],
                     block_conds: list[str], block_name: str) -> dict:
    """Compute within-block RSA for a subset of conditions."""
    # Get indices for block conditions in both RDMs
    llm_idx = [unique_conds.index(c) for c in block_conds]
    brain_idx = [brain_conds.index(c) for c in block_conds]

    llm_sub = rdm[np.ix_(llm_idx, llm_idx)]
    brain_sub = brain_rdm[np.ix_(brain_idx, brain_idx)]

    llm_tri = triu_vals(llm_sub)
    brain_tri = triu_vals(brain_sub)

    if len(llm_tri) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n_pairs": len(llm_tri)}

    rho, p = spearmanr(llm_tri, brain_tri)

    # Permutation p-value
    p_perm = permutation_pvalue(brain_tri, llm_tri, n_perm=10000, seed=42)

    return {
        "rho": round(float(rho), 4),
        "p_parametric": round(float(p), 5),
        "p_perm": round(float(p_perm), 5),
        "n_conditions": len(block_conds),
        "n_pairs": len(llm_tri),
        "block_name": block_name,
    }


def layer_sweep(per_stim_states: np.ndarray, conditions: list[str],
                unique_conds: list[str], brain_tri: np.ndarray) -> list[float]:
    """Compute RSA at every layer."""
    n_layers = per_stim_states.shape[1]
    rhos = []
    for L in range(n_layers):
        rdm = build_condition_rdm(per_stim_states, conditions, unique_conds, L)
        r, _ = spearmanr(triu_vals(rdm), brain_tri)
        rhos.append(round(float(r), 4))
    return rhos


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Template-matched RSA V2: extraction + analysis (one model)")
    parser.add_argument("--model_path", required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--model_short", required=True,
                        help="Short model name (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for extraction (default: {BATCH_SIZE})")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max token length for tokenization (default: 256)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    t0 = time.time()
    model_short = args.model_short

    # Determine peak layer
    if model_short in MODELS:
        peak_layer = MODELS[model_short]
    else:
        print(f"WARNING: {model_short} not in MODELS config, using layer 15 as default")
        peak_layer = 15

    print("=" * 70)
    print(f"Template-Matched RSA V2: {model_short}")
    print(f"Peak layer: {peak_layer}")
    print("=" * 70)

    # ── Load stimuli ──
    if not STIMULI_PATH.exists():
        print(f"ERROR: Stimuli file not found: {STIMULI_PATH}")
        print("Run template_matched_stimuli_v2.py first to generate stimuli.")
        return

    stimuli = load_stimuli(STIMULI_PATH)
    texts = [s["text"] for s in stimuli]
    conditions = [s["condition"] for s in stimuli]
    templates = [s["template"] for s in stimuli]
    unique_conds = sorted(set(conditions))
    print(f"Loaded {len(stimuli)} stimuli, {len(unique_conds)} conditions")

    # ── Load brain RDM ──
    brain_path = RSA_DIR / "brain_rdm.npz"
    if not brain_path.exists():
        print(f"ERROR: Brain RDM not found: {brain_path}")
        return
    brain = np.load(brain_path, allow_pickle=True)
    brain_conds = list(brain["conditions"])
    # Reorder brain RDM to match sorted condition order
    order = [brain_conds.index(c) for c in unique_conds]
    brain_rdm_full = brain["rdm"][np.ix_(order, order)]
    brain_tri = triu_vals(brain_rdm_full)
    print(f"Brain RDM: {len(brain_conds)} conditions loaded")

    # ── Load model ──
    print(f"\nLoading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    # ── Extract hidden states ──
    print(f"\nExtracting hidden states (batch_size={args.batch_size}) ...")
    t_extract = time.time()
    per_stim_states = extract_hidden_states(
        model, tokenizer, texts,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Extraction done in {time.time() - t_extract:.0f}s")
    print(f"Shape: {per_stim_states.shape}")  # [840, n_layers+1, hidden]

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory freed.")

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    # 1. Build headline RDM at peak layer
    rdm = build_condition_rdm(per_stim_states, conditions, unique_conds, peak_layer)
    rho, _ = spearmanr(triu_vals(rdm), brain_tri)
    print(f"\nHeadline RSA at peak layer {peak_layer}: rho = {rho:+.4f}")

    # 2. Permutation p-value
    p_perm = permutation_pvalue(brain_tri, triu_vals(rdm))
    print(f"Permutation p-value (10000 perms): {p_perm:.5f}")

    # 3. Split-half ceiling
    ceiling = split_half_ceiling(per_stim_states, conditions, unique_conds,
                                 peak_layer)
    print(f"Split-half ceiling: {ceiling:.4f}")
    if ceiling > 0:
        print(f"rho / ceiling: {rho / ceiling:.3f}")

    # 4. Per-template sub-RSA
    tmpl_rsa = per_template_rsa(per_stim_states, templates, conditions,
                                unique_conds, brain_tri, peak_layer)
    print(f"\nPer-template RSA:")
    for tmpl, tr in sorted(tmpl_rsa.items()):
        print(f"  {tmpl:15s}: rho = {tr:+.4f}")

    # 5. Within-block RSA
    print(f"\nWithin-block RSA:")
    within_aff = within_block_rsa(rdm, unique_conds, brain_rdm_full,
                                  unique_conds, WITHIN_AFFECTIVE,
                                  "within_affective_excl_valence")
    print(f"  Within-affective (excl valence, {within_aff['n_conditions']} conds, "
          f"{within_aff['n_pairs']} pairs): rho = {within_aff['rho']:+.4f}, "
          f"p_perm = {within_aff['p_perm']:.4f}")

    within_soc = within_block_rsa(rdm, unique_conds, brain_rdm_full,
                                  unique_conds, WITHIN_SOCIAL,
                                  "within_social_excl_moral")
    print(f"  Within-social (excl moral, {within_soc['n_conditions']} conds, "
          f"{within_soc['n_pairs']} pairs): rho = {within_soc['rho']:+.4f}, "
          f"p_perm = {within_soc['p_perm']:.4f}")

    # 6. Layer sweep
    print(f"\nLayer sweep ...")
    layer_rhos = layer_sweep(per_stim_states, conditions, unique_conds, brain_tri)
    actual_peak = int(np.argmax(layer_rhos))
    print(f"  Actual peak layer: {actual_peak} (rho = {layer_rhos[actual_peak]:+.4f})")
    print(f"  Headline peak layer {peak_layer}: rho = {layer_rhos[peak_layer]:+.4f}")

    # 7. Load V1 results for comparison
    v1_comparison = {}
    v1_results_path = V1_DIR / "template_matched_rsa_results.json"
    if v1_results_path.exists():
        with open(v1_results_path) as f:
            v1_data = json.load(f)
        if model_short in v1_data.get("models", {}):
            v1_r = v1_data["models"][model_short]
            v1_comparison = {
                "v1_rho_at_headline_peak": v1_r["rho_at_headline_peak"],
                "v1_ceiling": v1_r["ceiling"],
                "v1_original_stimuli_rho": v1_r.get("original_stimuli_rho"),
            }
            print(f"\nV1 comparison:")
            print(f"  V1 rho at headline peak: {v1_r['rho_at_headline_peak']:+.4f}")
            print(f"  V2 rho at headline peak: {rho:+.4f}")
            print(f"  Delta (V2 - V1): {rho - v1_r['rho_at_headline_peak']:+.4f}")
            if v1_r.get("original_stimuli_rho"):
                print(f"  Original stimuli rho: {v1_r['original_stimuli_rho']:+.4f}")

    # Load original headline RDM for comparison
    orig_rho = None
    orig_rdm_path = RSA_DIR / f"{model_short}_rdm14_headline.npz"
    if orig_rdm_path.exists():
        z = np.load(orig_rdm_path, allow_pickle=True)
        orig_rdm = z["rdm"]
        orig_conds = list(z["conditions"])
        orig_order = [brain_conds.index(c) for c in orig_conds]
        brain_re = brain["rdm"][np.ix_(orig_order, orig_order)]
        orig_rho_val, _ = spearmanr(triu_vals(orig_rdm), triu_vals(brain_re))
        orig_rho = round(float(orig_rho_val), 4)
        print(f"\n  Original (diverse-source) stimuli rho: {orig_rho:+.4f}")
        print(f"  V2 template-matched rho: {rho:+.4f}")
        print(f"  Retention: {rho / orig_rho_val * 100:.1f}%")

    # ── Save results ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save RDM
    np.savez_compressed(
        OUT_DIR / f"{model_short}_tm_v2_rdm14.npz",
        rdm=rdm,
        conditions=np.array(unique_conds),
        peak_layer=peak_layer,
        config="mean_all|centered|1_cosine|template_matched_v2",
    )
    print(f"\nSaved RDM to {OUT_DIR / f'{model_short}_tm_v2_rdm14.npz'}")

    # Save JSON results
    results = {
        "model": model_short,
        "model_path": args.model_path,
        "config": "mean_all|centered|1_cosine|template_matched_v2",
        "peak_layer": peak_layer,
        "rho_at_headline_peak": round(float(rho), 4),
        "p_perm": round(float(p_perm), 5),
        "ceiling": round(float(ceiling), 4),
        "rho_over_ceiling": round(float(rho) / ceiling, 3) if ceiling > 0 else None,
        "actual_peak_layer": actual_peak,
        "rho_at_actual_peak": layer_rhos[actual_peak],
        "per_template_rho": tmpl_rsa,
        "within_affective": within_aff,
        "within_social": within_soc,
        "layer_rhos": layer_rhos,
        "original_stimuli_rho": orig_rho,
        "v1_comparison": v1_comparison,
        "n_stimuli": len(stimuli),
        "n_conditions": len(unique_conds),
        "extraction_time_s": round(time.time() - t_extract, 0),
        "total_time_s": round(time.time() - t0, 0),
    }

    results_path = OUT_DIR / f"{model_short}_tm_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # ── Final summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Model:             {model_short}")
    print(f"V2 TM rho:         {rho:+.4f} (p_perm = {p_perm:.4f})")
    print(f"Ceiling:           {ceiling:.4f}")
    print(f"rho/ceiling:       {rho/ceiling:.3f}" if ceiling > 0 else "rho/ceiling: N/A")
    if orig_rho:
        print(f"Original rho:      {orig_rho:+.4f}")
        print(f"Retention:         {rho / orig_rho * 100:.1f}%")
    if v1_comparison:
        v1_rho = v1_comparison["v1_rho_at_headline_peak"]
        print(f"V1 TM rho:         {v1_rho:+.4f}")
        print(f"V2 - V1 delta:     {rho - v1_rho:+.4f}")
    print(f"Within-affective:  {within_aff['rho']:+.4f} (p_perm = {within_aff['p_perm']:.4f})")
    print(f"Within-social:     {within_soc['rho']:+.4f} (p_perm = {within_soc['p_perm']:.4f})")
    print(f"Peak layer (sweep): L{actual_peak} (rho = {layer_rhos[actual_peak]:+.4f})")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
