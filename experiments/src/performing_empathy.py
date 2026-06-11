#!/usr/bin/env python3
"""
Performing Empathy: Do LLMs have genuine emotional differentiation
or just surface-level behavioral acting?

Design:
  Phase 1 — Identify condition-selective neurons from original 14-condition RSA stimuli
             (activation-based, no gradients → fast)
  Phase 2 — Process 50 emotional support-seeking prompts (10 scenarios × 5 emotions):
             (a) extract hidden states → within-emotion internal RDM
             (b) record gate activations at selective neuron positions → condition profile
             (c) generate comfort responses → behavioral RDM (TF-IDF)
  Phase 3 — Analysis:
             - Internal vs behavioral dissociation (t-test on within-emotion distances)
             - Neuron activation profiling (which cognitive system activates?)
             - Brain comparison (internal RDM vs brain within-affective RDM)

GPU required. One model per run.
Usage:
  python src/performing_empathy.py --model_path /path/to/model [--model_short name]
"""
from __future__ import annotations
import argparse, json, gc, sys, time
import numpy as np
import torch
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr, mannwhitneyu, ttest_rel
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

CONDITIONS_14 = [
    "anger", "belief", "disgust", "empathy", "fear", "happiness",
    "intention", "judgment", "mentalizing", "moral", "sadness",
    "self_referential", "theory_of_mind", "valence",
]
EMOTIONS_5 = ["anger", "disgust", "fear", "happiness", "sadness"]
SOCIAL_8 = ["belief", "intention", "judgment", "mentalizing", "moral",
            "empathy", "self_referential", "theory_of_mind"]

PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}

SCENARIOS = [
    "My colleague took credit for my work in front of the boss today",
    "I found out my partner has been lying to me for months",
    "My project that I worked on for a year was suddenly cancelled",
    "I made a serious mistake at work and it affected the whole team",
    "My close friend stopped talking to me without any explanation",
    "I got passed over for a promotion despite working harder than anyone",
    "Someone I trusted shared my personal secrets with others",
    "I received harsh public criticism on something I put months of effort into",
    "I was deliberately excluded from an important meeting by my manager",
    "My presentation went terribly wrong in front of important clients",
]

EMOTION_PHRASES = {
    "anger": "furious and full of rage",
    "disgust": "repulsed and sickened by the whole situation",
    "fear": "terrified about what this means for my future",
    "happiness": "strangely relieved, almost glad it happened because now I see things clearly",
    "sadness": "heartbroken and unable to stop crying",
}

N_SELECTIVE = 5000


# ═══════════════════════════════════════════════════════════════════════
# Model utilities
# ═══════════════════════════════════════════════════════════════════════

def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find model layers")


def get_ffn_dim(model):
    layers = get_model_layers(model)
    mlp = layers[0].mlp
    if hasattr(mlp, "gate_proj"):
        return mlp.gate_proj.out_features
    raise ValueError("Cannot determine FFN dim")


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Identify selective neurons (activation-based, no gradients)
# ═══════════════════════════════════════════════════════════════════════

def identify_selective_neurons(model, tokenizer, stimuli, device,
                               n_layers, ffn_dim, batch_size=8):
    """Forward-pass activation profiling per condition → one-vs-rest selectivity."""
    n_neurons = n_layers * ffn_dim
    cond_act = {c: np.zeros(n_neurons, dtype=np.float64) for c in CONDITIONS_14}
    cond_n = {c: 0 for c in CONDITIONS_14}

    layers = get_model_layers(model)
    gate_acts = {}
    hooks = []

    for lidx in range(n_layers):
        mlp = layers[lidx].mlp
        target = mlp.gate_proj

        def make_hook(idx):
            def fwd_hook(module, inp, out):
                gate_acts[idx] = out.detach().float()
            return fwd_hook
        hooks.append(target.register_forward_hook(make_hook(lidx)))

    print(f"  Profiling {len(stimuli)} stimuli for neuron selectivity...")

    for i in range(0, len(stimuli), batch_size):
        batch = stimuli[i:i+batch_size]
        texts = [s["text"] for s in batch]
        conds = [s["condition"] for s in batch]

        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
        gate_acts.clear()

        with torch.no_grad():
            model(**inputs)

        mask = inputs["attention_mask"]  # (B, seq)
        for lidx in range(n_layers):
            if lidx not in gate_acts:
                continue
            act = gate_acts[lidx]  # (B, seq, ffn_dim)
            act_abs = act.abs()
            mask_3d = mask.unsqueeze(-1).to(act_abs.dtype)
            mean_act = (act_abs * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp(min=1)

            start = lidx * ffn_dim
            end = start + ffn_dim
            for bi in range(len(batch)):
                c = conds[bi]
                cond_act[c][start:end] += mean_act[bi].cpu().numpy().astype(np.float64)

        for bi in range(len(batch)):
            cond_n[conds[bi]] += 1

        if (i // batch_size + 1) % 20 == 0:
            print(f"    batch {i // batch_size + 1}/{(len(stimuli) + batch_size - 1) // batch_size}")

    for h in hooks:
        h.remove()

    for c in CONDITIONS_14:
        if cond_n[c] > 0:
            cond_act[c] /= cond_n[c]

    # One-vs-rest selectivity
    selectivity = {}
    for target in CONDITIONS_14:
        others = np.mean([cond_act[c] for c in CONDITIONS_14 if c != target], axis=0)
        sel = (cond_act[target] - others) / (cond_act[target] + others + 1e-10)
        selectivity[target] = sel

    # Top-N indices per condition
    neuron_indices = {}
    for c in CONDITIONS_14:
        top_idx = np.argsort(-selectivity[c])[:N_SELECTIVE]
        neuron_indices[c] = top_idx
        mean_sel = selectivity[c][top_idx].mean()
        print(f"    {c:>20s}: top-{N_SELECTIVE} mean sel = {mean_sel:.4f}")

    return selectivity, neuron_indices


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Process emotional support prompts
# ═══════════════════════════════════════════════════════════════════════

def build_support_prompts():
    """10 scenarios × 5 emotions = 50 prompts."""
    prompts = []
    for si, scenario in enumerate(SCENARIOS):
        for emo in EMOTIONS_5:
            text = f"{scenario}. I feel {EMOTION_PHRASES[emo]}. I really need someone to talk to."
            prompts.append({
                "text": text,
                "emotion": emo,
                "scenario_idx": si,
            })
    return prompts


def extract_support_representations(model, tokenizer, prompts, device,
                                     n_layers, ffn_dim, neuron_indices,
                                     peak_layer):
    """Extract hidden states + gate activations at selective neuron positions."""
    layers = get_model_layers(model)
    gate_acts = {}
    hooks = []

    for lidx in range(n_layers):
        def make_hook(idx):
            def fwd_hook(module, inp, out):
                gate_acts[idx] = out.detach().float()
            return fwd_hook
        hooks.append(layers[lidx].mlp.gate_proj.register_forward_hook(make_hook(lidx)))

    hidden_states = []
    condition_profiles = []  # per-prompt: activation at each condition's neurons

    print(f"  Extracting representations for {len(prompts)} support prompts...")

    for pi, p in enumerate(prompts):
        inputs = tokenizer(p["text"], return_tensors="pt", truncation=True,
                          max_length=512).to(device)
        gate_acts.clear()

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # Hidden state at peak layer (mean-pooled)
        h = out.hidden_states[peak_layer][0].float()  # (seq, hidden)
        mask = inputs["attention_mask"][0].bool()
        hidden_states.append(h[mask].mean(dim=0).cpu().numpy())

        # Gate activation profile: mean |activation| at each condition's selective neurons
        profile = {}
        for cond, idx in neuron_indices.items():
            total_act = 0.0
            count = 0
            for n in idx:
                l = int(n // ffn_dim)
                pos = int(n % ffn_dim)
                if l in gate_acts:
                    act_val = gate_acts[l][0, :, pos].abs().mean().item()
                    total_act += act_val
                    count += 1
            profile[cond] = total_act / max(count, 1)
        condition_profiles.append(profile)

        if (pi + 1) % 10 == 0:
            print(f"    {pi+1}/{len(prompts)}")

    for h in hooks:
        h.remove()

    return np.array(hidden_states), condition_profiles


def generate_responses(model, tokenizer, prompts, device, max_new=200):
    """Generate comfort responses for each prompt."""
    responses = []
    print(f"  Generating responses for {len(prompts)} prompts...")

    for pi, p in enumerate(prompts):
        messages = [{"role": "user", "content": p["text"]}]

        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_text = f"User: {p['text']}\nAssistant:"

        inputs = tokenizer(chat_text, return_tensors="pt", truncation=True,
                          max_length=512).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(resp)

        if (pi + 1) % 10 == 0:
            print(f"    {pi+1}/{len(prompts)}")

    return responses


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_rdm_from_centroids(vectors, labels, conditions):
    """Compute condition centroids → pairwise cosine distance → RDM."""
    centroids = {}
    for c in conditions:
        mask = [i for i, l in enumerate(labels) if l == c]
        if mask:
            centroids[c] = np.mean(vectors[mask], axis=0)

    grand_mean = np.mean(list(centroids.values()), axis=0)
    for c in centroids:
        centroids[c] = centroids[c] - grand_mean

    pairs = list(combinations(conditions, 2))
    dists = []
    for c1, c2 in pairs:
        v1, v2 = centroids[c1], centroids[c2]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        dists.append(1 - cos)
    return pairs, dists


def compute_behavioral_rdm(responses, labels, conditions):
    """TF-IDF cosine similarity between response centroids per emotion."""
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    vecs = tfidf.fit_transform(responses).toarray()

    centroids = {}
    for c in conditions:
        mask = [i for i, l in enumerate(labels) if l == c]
        if mask:
            centroids[c] = np.mean(vecs[mask], axis=0)

    pairs = list(combinations(conditions, 2))
    dists = []
    for c1, c2 in pairs:
        cos = cosine_similarity([centroids[c1]], [centroids[c2]])[0, 0]
        dists.append(1 - cos)
    return pairs, dists


def analyze_neuron_profiles(profiles, prompts):
    """Which cognitive system activates most for support prompts?"""
    emo_labels = [p["emotion"] for p in prompts]

    # Mean activation per condition's neurons, averaged across all prompts
    all_cond_means = {c: np.mean([p[c] for p in profiles]) for c in CONDITIONS_14}

    # Block means
    aff_act = np.mean([all_cond_means[c] for c in EMOTIONS_5])
    soc_act = np.mean([all_cond_means[c] for c in SOCIAL_8])
    emp_act = all_cond_means["empathy"]

    # Per-emotion breakdown: when user says "I'm angry", which neurons light up?
    per_emo_profiles = {}
    for emo in EMOTIONS_5:
        mask = [i for i, e in enumerate(emo_labels) if e == emo]
        per_emo_profiles[emo] = {c: np.mean([profiles[i][c] for i in mask])
                                  for c in CONDITIONS_14}

    # Key test: does the MATCHING emotion neuron activate more than others?
    match_vs_mismatch = []
    for emo in EMOTIONS_5:
        match = per_emo_profiles[emo][emo]
        others = [per_emo_profiles[emo][e] for e in EMOTIONS_5 if e != emo]
        mismatch = np.mean(others)
        match_vs_mismatch.append({
            "emotion": emo,
            "match_activation": float(match),
            "mismatch_activation": float(mismatch),
            "ratio": float(match / mismatch) if mismatch > 0 else float("inf"),
        })

    # Empathy neuron activation across all emotion prompts
    emp_by_emo = {emo: per_emo_profiles[emo]["empathy"] for emo in EMOTIONS_5}

    return {
        "overall_condition_means": {c: float(v) for c, v in all_cond_means.items()},
        "block_means": {
            "affective_neurons": float(aff_act),
            "social_neurons": float(soc_act),
            "empathy_neurons": float(emp_act),
        },
        "match_vs_mismatch": match_vs_mismatch,
        "empathy_by_emotion": {k: float(v) for k, v in emp_by_emo.items()},
        "per_emotion_profiles": {
            emo: {c: float(v) for c, v in prof.items()}
            for emo, prof in per_emo_profiles.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peak = PEAK.get(model_short, 15)

    print(f"{'='*70}")
    print(f"PERFORMING EMPATHY: {model_short}")
    print(f"{'='*70}")
    print(f"Device: {device}, Peak layer: L{peak}")

    # Load brain within-aff distances
    brain = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    brain_conds = list(brain["conditions"])
    brain_rdm = brain["rdm"]
    brain_aff_pairs = list(combinations(EMOTIONS_5, 2))
    brain_aff_dists = [brain_rdm[brain_conds.index(c1), brain_conds.index(c2)]
                       for c1, c2 in brain_aff_pairs]

    # Load original RSA stimuli
    stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    with open(stim_path) as f:
        rsa_stimuli = [json.loads(l) for l in f]
    n_conds = len(set(s["condition"] for s in rsa_stimuli))
    print(f"Loaded {len(rsa_stimuli)} RSA stimuli ({n_conds} conditions)")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    n_layers = len(get_model_layers(model))
    ffn_dim = get_ffn_dim(model)
    print(f"  {n_layers} layers, ffn_dim={ffn_dim}, total neurons={n_layers*ffn_dim}")

    # ═══ PHASE 1: Identify selective neurons ═══
    t0 = time.time()
    print(f"\n{'='*70}")
    print("PHASE 1: Identify condition-selective neurons")
    print(f"{'='*70}")
    selectivity, neuron_indices = identify_selective_neurons(
        model, tokenizer, rsa_stimuli, device, n_layers, ffn_dim)
    print(f"  Phase 1 done ({time.time()-t0:.0f}s)")

    # ═══ PHASE 2: Process support prompts ═══
    prompts = build_support_prompts()
    print(f"\n{'='*70}")
    print(f"PHASE 2: Process {len(prompts)} support prompts")
    print(f"{'='*70}")

    # 2a: Hidden states + gate activation profiles
    t1 = time.time()
    hidden_states, condition_profiles = extract_support_representations(
        model, tokenizer, prompts, device, n_layers, ffn_dim,
        neuron_indices, peak)
    print(f"  Extraction done ({time.time()-t1:.0f}s)")

    # 2b: Generate responses
    t2 = time.time()
    responses = generate_responses(model, tokenizer, prompts, device)
    print(f"  Generation done ({time.time()-t2:.0f}s)")

    # ═══ PHASE 3: Analysis ═══
    print(f"\n{'='*70}")
    print("PHASE 3: Analysis")
    print(f"{'='*70}")

    emo_labels = [p["emotion"] for p in prompts]

    # 3a: Internal RDM (hidden states)
    int_pairs, int_dists = compute_rdm_from_centroids(
        hidden_states, emo_labels, EMOTIONS_5)
    int_rho, int_p = spearmanr(brain_aff_dists, int_dists)
    print(f"\n  INTERNAL RDM (hidden states @ L{peak}):")
    print(f"    vs brain: ρ = {int_rho:+.3f} (p = {int_p:.4f})")
    int_mean_dist = float(np.mean(int_dists))
    print(f"    mean within-aff distance: {int_mean_dist:.4f}")

    # 3b: Behavioral RDM (responses)
    beh_pairs, beh_dists = compute_behavioral_rdm(
        responses, emo_labels, EMOTIONS_5)
    beh_rho, beh_p = spearmanr(brain_aff_dists, beh_dists)
    print(f"\n  BEHAVIORAL RDM (TF-IDF response similarity):")
    print(f"    vs brain: ρ = {beh_rho:+.3f} (p = {beh_p:.4f})")
    beh_mean_dist = float(np.mean(beh_dists))
    print(f"    mean within-emotion distance: {beh_mean_dist:.4f}")

    # 3c: Dissociation test
    print(f"\n  DISSOCIATION (behavioral vs internal):")
    print(f"    Internal mean dist:   {int_mean_dist:.4f}")
    print(f"    Behavioral mean dist: {beh_mean_dist:.4f}")
    if len(int_dists) == len(beh_dists):
        t_stat, t_p = ttest_rel(beh_dists, int_dists)
        print(f"    Paired t-test: t = {t_stat:.3f}, p = {t_p:.4f}")
        dissociation_sig = t_p < 0.05
    else:
        t_stat, t_p = float("nan"), float("nan")
        dissociation_sig = False

    # 3d: Neuron activation profiling
    print(f"\n  NEURON ACTIVATION PROFILING:")
    neuron_results = analyze_neuron_profiles(condition_profiles, prompts)

    print(f"    Block activation (support prompts):")
    bm = neuron_results["block_means"]
    print(f"      Affective neurons:  {bm['affective_neurons']:.4f}")
    print(f"      Social neurons:     {bm['social_neurons']:.4f}")
    print(f"      Empathy neurons:    {bm['empathy_neurons']:.4f}")

    print(f"\n    Match vs Mismatch (does 'angry' prompt activate anger neurons more?):")
    for m in neuron_results["match_vs_mismatch"]:
        print(f"      {m['emotion']:>10s}: match={m['match_activation']:.4f}, "
              f"mismatch={m['mismatch_activation']:.4f}, ratio={m['ratio']:.3f}")

    # Mean match ratio across emotions
    match_ratios = [m["ratio"] for m in neuron_results["match_vs_mismatch"]]
    mean_ratio = float(np.mean(match_ratios))
    print(f"    Mean match/mismatch ratio: {mean_ratio:.3f}")
    print(f"    Interpretation: ratio ≈ 1.0 → no specificity (model treats all emotions same)")
    print(f"                    ratio >> 1.0 → specific (model activates matching neurons)")

    print(f"\n    Empathy neuron activation by user emotion:")
    for emo, val in neuron_results["empathy_by_emotion"].items():
        print(f"      {emo:>10s}: {val:.4f}")
    emp_vals = list(neuron_results["empathy_by_emotion"].values())
    emp_cv = float(np.std(emp_vals) / np.mean(emp_vals)) if np.mean(emp_vals) > 0 else 0
    print(f"    CV across emotions: {emp_cv:.3f} (low CV = empathy doesn't care which emotion)")

    # 3e: Per-scenario response examples
    print(f"\n  RESPONSE EXAMPLES (first 3 scenarios):")
    for si in range(min(3, len(SCENARIOS))):
        print(f"\n    Scenario {si}: {SCENARIOS[si][:60]}...")
        for emo in EMOTIONS_5:
            idx = si * len(EMOTIONS_5) + EMOTIONS_5.index(emo)
            resp_preview = responses[idx][:100].replace("\n", " ")
            print(f"      [{emo:>10s}]: {resp_preview}...")

    # ═══ SUMMARY ═══
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_short}")
    print(f"{'='*70}")
    print(f"  Internal RDM vs brain:    ρ = {int_rho:+.3f} (p = {int_p:.4f})")
    print(f"  Behavioral RDM vs brain:  ρ = {beh_rho:+.3f} (p = {beh_p:.4f})")
    print(f"  Dissociation (beh > int): t = {t_stat:.3f}, p = {t_p:.4f}")
    print(f"  Neuron match ratio:       {mean_ratio:.3f}")
    print(f"  Empathy neuron CV:        {emp_cv:.3f}")
    print()

    verdict = []
    if int_rho < 0.3:
        verdict.append("Internal representations DO NOT differentiate emotions (compressed)")
    if beh_mean_dist > int_mean_dist * 1.5:
        verdict.append("Behavioral output differentiates MORE than internal state")
    if mean_ratio < 1.1:
        verdict.append("Emotion-specific neurons NOT selectively activated")
    if emp_cv < 0.1:
        verdict.append("Empathy neurons fire uniformly regardless of emotion type")

    if verdict:
        print("  VERDICT: PERFORMING EMPATHY")
        for v in verdict:
            print(f"    → {v}")
    else:
        print("  VERDICT: Genuine differentiation detected (unexpected)")

    # ═══ SAVE ═══
    results = {
        "model": model_short,
        "peak_layer": peak,
        "n_prompts": len(prompts),
        "n_scenarios": len(SCENARIOS),
        "n_emotions": len(EMOTIONS_5),
        "internal_rdm": {
            "rho_vs_brain": float(int_rho),
            "p_vs_brain": float(int_p),
            "mean_distance": int_mean_dist,
            "pairs": [f"{c1}-{c2}" for c1, c2 in int_pairs],
            "distances": [float(d) for d in int_dists],
        },
        "behavioral_rdm": {
            "rho_vs_brain": float(beh_rho),
            "p_vs_brain": float(beh_p),
            "mean_distance": beh_mean_dist,
            "pairs": [f"{c1}-{c2}" for c1, c2 in beh_pairs],
            "distances": [float(d) for d in beh_dists],
        },
        "dissociation": {
            "t_stat": float(t_stat),
            "p_value": float(t_p),
            "behavioral_gt_internal": bool(beh_mean_dist > int_mean_dist),
        },
        "neuron_profiling": neuron_results,
        "neuron_match_ratio": mean_ratio,
        "empathy_cv": emp_cv,
        "responses": [
            {"scenario": prompts[i]["scenario_idx"],
             "emotion": prompts[i]["emotion"],
             "response": responses[i][:500]}
            for i in range(len(prompts))
        ],
        "verdict": verdict,
    }

    out_path = OUT / f"performing_empathy_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nSaved: {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
