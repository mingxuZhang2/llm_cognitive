#!/usr/bin/env python3
"""
Label-Context Conflict RSA: extraction + causal analysis.

Tests the causal direction of emotion representation: when emotion label and
situational context CONFLICT, does the model follow the label or the context?

Key finding context: within-affective RSA is rho=-0.11 under short stimuli but
rho=+0.81 under template-matched (richer) stimuli. This suggests emotion
representations are input-driven. This experiment tests the causal direction.

Pipeline:
  Phase 1: Extract hidden states at peak layer (mean_all pooling) + all layers
  Phase 2: Compute emotion centroids from label_only condition
  Phase 3: Analyze context_only condition (can model identify emotions from
           context alone?)
  Phase 4: THE KEY TEST -- conflict condition: is representation closer to
           LABEL centroid or CONTEXT centroid?
  Phase 5: Build 5x5 RDMs per condition, compare to brain within-affective RDM
  Phase 6: Behavioral test -- generate responses to conflict items, check
           whether response addresses label emotion or context situation

GPU required. One model per run.
Usage:
  python src/label_context_conflict_rsa.py --model_path /path/to/model \
    --model_short Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
STIM_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "label_context_conflict_stimuli.jsonl"
OUT_DIR = BASE / "results" / "label_context_conflict"

MODELS_PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}

# 14 conditions in brain_rdm.npz order
CONDITIONS_14 = [
    "anger", "belief", "disgust", "empathy", "fear", "happiness",
    "intention", "judgment", "mentalizing", "moral", "sadness",
    "self_referential", "theory_of_mind", "valence",
]

# The 5 affective conditions used in this experiment and their indices in
# the 14-condition brain RDM
EMOTIONS_5 = ["anger", "disgust", "fear", "happiness", "sadness"]
BRAIN_INDICES_5 = [0, 2, 4, 5, 10]  # anger=0, disgust=2, fear=4, happiness=5, sadness=10

# Conflict pairings: label_emotion -> context_emotion
CONFLICT_MAP = {
    "anger":     "sadness",
    "sadness":   "anger",
    "fear":      "happiness",
    "disgust":   "fear",
    "happiness": "disgust",
}


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find model layers")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return 1.0 - dot / (na * nb)


def rdm_cosine(act: np.ndarray) -> np.ndarray:
    """Compute cosine-distance RDM from condition centroids."""
    n = np.linalg.norm(act, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = act / n
    return 1.0 - Xn @ Xn.T


def triu_vals(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def load_brain_affective_rdm() -> tuple[np.ndarray, list[str]]:
    """Load brain RDM and extract the 5 affective conditions sub-RDM."""
    brain = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain["rdm"]
    brain_conds = list(brain["conditions"])

    # Extract 5x5 sub-RDM for the 5 emotions
    idx = [brain_conds.index(e) for e in EMOTIONS_5]
    sub_rdm = brain_rdm[np.ix_(idx, idx)]
    return sub_rdm, EMOTIONS_5


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Extract hidden states
# ═══════════════════════════════════════════════════════════════════════

def extract_hidden_states(model, tokenizer, stimuli, device, n_layers, peak_layer):
    """Extract hidden states: mean_all pooling at peak layer + all layers."""
    print(f"  Extracting hidden states for {len(stimuli)} stimuli...")
    peak_states = []       # (n_stimuli, hidden_dim) at peak layer
    all_layer_states = []  # (n_stimuli, n_layers, hidden_dim)

    for si, s in enumerate(stimuli):
        inputs = tokenizer(s["text"], return_tensors="pt", truncation=True,
                           max_length=512).to(device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        mask = inputs["attention_mask"][0].bool()

        # Peak layer: mean_all pooling
        h_peak = out.hidden_states[peak_layer][0].float()  # (seq, hidden)
        h_peak_mean = h_peak[mask].mean(dim=0).cpu().numpy()
        peak_states.append(h_peak_mean)

        # All layers: mean_all pooling
        layer_vecs = []
        for L in range(n_layers + 1):  # +1 for embedding layer
            h_L = out.hidden_states[L][0].float()
            h_L_mean = h_L[mask].mean(dim=0).cpu().numpy()
            layer_vecs.append(h_L_mean)
        all_layer_states.append(layer_vecs)

        if (si + 1) % 25 == 0:
            print(f"    {si + 1}/{len(stimuli)}")

    peak_states = np.array(peak_states)
    all_layer_states = np.array(all_layer_states)  # (n_stim, n_layers+1, hidden)

    return peak_states, all_layer_states


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Compute emotion centroids from label_only
# ═══════════════════════════════════════════════════════════════════════

def compute_centroids(peak_states, stimuli, target_condition="label_only"):
    """Compute per-emotion centroid from the specified condition."""
    centroids = {}
    for emo in EMOTIONS_5:
        idx = [i for i, s in enumerate(stimuli)
               if s["condition"] == target_condition and s["emotion"] == emo]
        centroids[emo] = peak_states[idx].mean(axis=0)
    return centroids


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Context-only accuracy
# ═══════════════════════════════════════════════════════════════════════

def analyze_context_only(peak_states, stimuli, centroids):
    """For each context_only item, is the nearest centroid the correct emotion?"""
    results = {"correct": 0, "total": 0, "per_emotion": {}, "confusions": []}

    for emo in EMOTIONS_5:
        results["per_emotion"][emo] = {"correct": 0, "total": 0}

    for i, s in enumerate(stimuli):
        if s["condition"] != "context_only":
            continue

        vec = peak_states[i]
        dists = {e: cosine_distance(vec, centroids[e]) for e in EMOTIONS_5}
        nearest = min(dists, key=dists.get)
        correct = (nearest == s["emotion"])

        results["total"] += 1
        results["per_emotion"][s["emotion"]]["total"] += 1
        if correct:
            results["correct"] += 1
            results["per_emotion"][s["emotion"]]["correct"] += 1
        else:
            results["confusions"].append({
                "item_id": s["item_id"],
                "true_emotion": s["emotion"],
                "predicted": nearest,
                "distances": {e: round(float(d), 5) for e, d in dists.items()},
            })

    results["accuracy"] = results["correct"] / max(results["total"], 1)
    for emo in EMOTIONS_5:
        e = results["per_emotion"][emo]
        e["accuracy"] = e["correct"] / max(e["total"], 1)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Conflict analysis (THE KEY TEST)
# ═══════════════════════════════════════════════════════════════════════

def analyze_conflict(peak_states, stimuli, centroids):
    """For conflict items: closer to LABEL centroid or CONTEXT centroid?"""
    label_wins = 0
    context_wins = 0
    total = 0
    per_emotion = {e: {"label_wins": 0, "context_wins": 0, "total": 0}
                   for e in EMOTIONS_5}
    details = []

    for i, s in enumerate(stimuli):
        if s["condition"] != "conflict":
            continue

        label_emo = s["emotion"]
        ctx_emo = s["conflict_context_emotion"]
        vec = peak_states[i]

        d_label = cosine_distance(vec, centroids[label_emo])
        d_context = cosine_distance(vec, centroids[ctx_emo])

        # Also compute distances to ALL centroids
        all_dists = {e: cosine_distance(vec, centroids[e]) for e in EMOTIONS_5}
        nearest = min(all_dists, key=all_dists.get)

        is_label = d_label < d_context
        total += 1
        per_emotion[label_emo]["total"] += 1

        if is_label:
            label_wins += 1
            per_emotion[label_emo]["label_wins"] += 1
        else:
            context_wins += 1
            per_emotion[label_emo]["context_wins"] += 1

        details.append({
            "item_id": s["item_id"],
            "label_emotion": label_emo,
            "context_emotion": ctx_emo,
            "d_label": round(float(d_label), 5),
            "d_context": round(float(d_context), 5),
            "winner": "label" if is_label else "context",
            "nearest_centroid": nearest,
            "all_distances": {e: round(float(d), 5) for e, d in all_dists.items()},
        })

    # Per-emotion label rates
    per_emo_summary = {}
    for emo in EMOTIONS_5:
        pe = per_emotion[emo]
        pe["label_rate"] = pe["label_wins"] / max(pe["total"], 1)
        pe["context_rate"] = pe["context_wins"] / max(pe["total"], 1)
        per_emo_summary[emo] = {
            "label_rate": round(pe["label_rate"], 3),
            "context_rate": round(pe["context_rate"], 3),
            "n": pe["total"],
            "context_emotion": CONFLICT_MAP[emo],
        }

    return {
        "label_wins": label_wins,
        "context_wins": context_wins,
        "total": total,
        "label_rate": round(label_wins / max(total, 1), 3),
        "context_rate": round(context_wins / max(total, 1), 3),
        "per_emotion": per_emo_summary,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Condition-level RDMs + brain comparison
# ═══════════════════════════════════════════════════════════════════════

def build_condition_rdm(peak_states, stimuli, condition, emotions=None):
    """Build a 5x5 emotion RDM for a given condition (label_only or context_only)."""
    if emotions is None:
        emotions = EMOTIONS_5

    # Compute centroids
    centroids_list = []
    for emo in emotions:
        idx = [i for i, s in enumerate(stimuli)
               if s["condition"] == condition and s["emotion"] == emo]
        if not idx:
            centroids_list.append(np.zeros(peak_states.shape[1]))
        else:
            centroids_list.append(peak_states[idx].mean(axis=0))

    act = np.stack(centroids_list).astype(np.float64)
    # Center
    act = act - act.mean(axis=0, keepdims=True)
    # Cosine distance RDM
    rdm = rdm_cosine(act)
    return rdm


def compare_to_brain(rdm_5x5, brain_rdm_5x5):
    """Spearman rho between upper triangles of two 5x5 RDMs."""
    tri_model = triu_vals(rdm_5x5)
    tri_brain = triu_vals(brain_rdm_5x5)
    rho, p = spearmanr(tri_model, tri_brain)
    return float(rho), float(p)


# ═══════════════════════════════════════════════════════════════════════
# Phase 5b: Layer sweep for conflict
# ═══════════════════════════════════════════════════════════════════════

def layer_sweep_conflict(all_layer_states, stimuli, n_layers):
    """Compute conflict label_rate at each layer."""
    results = []

    for L in range(n_layers + 1):
        # Build label_only centroids at this layer
        centroids = {}
        for emo in EMOTIONS_5:
            idx = [i for i, s in enumerate(stimuli)
                   if s["condition"] == "label_only" and s["emotion"] == emo]
            centroids[emo] = all_layer_states[idx, L, :].mean(axis=0)

        label_wins = 0
        total = 0
        for i, s in enumerate(stimuli):
            if s["condition"] != "conflict":
                continue
            vec = all_layer_states[i, L, :]
            d_label = cosine_distance(vec, centroids[s["emotion"]])
            d_ctx = cosine_distance(vec, centroids[s["conflict_context_emotion"]])
            if d_label < d_ctx:
                label_wins += 1
            total += 1

        results.append(round(label_wins / max(total, 1), 3))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Behavioral test (generate responses to conflict items)
# ═══════════════════════════════════════════════════════════════════════

def generate_and_analyze_behavioral(model, tokenizer, stimuli, peak_states,
                                     centroids, device):
    """Generate responses for conflict items and check label vs context alignment."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    # First, generate responses for label_only items (reference corpus)
    label_only_items = [s for s in stimuli if s["condition"] == "label_only"]
    conflict_items = [s for s in stimuli if s["condition"] == "conflict"]

    print(f"  Generating responses for {len(label_only_items)} label_only items (reference)...")
    label_responses = {}
    for emo in EMOTIONS_5:
        label_responses[emo] = []

    for s in label_only_items:
        resp = _generate_one(model, tokenizer, s["text"], device)
        label_responses[s["emotion"]].append(resp)

    print(f"  Generating responses for {len(conflict_items)} conflict items...")
    conflict_responses = []
    for s in conflict_items:
        resp = _generate_one(model, tokenizer, s["text"], device)
        conflict_responses.append(resp)

    # Build TF-IDF from all label_only responses
    all_ref_texts = []
    all_ref_labels = []
    for emo in EMOTIONS_5:
        for resp in label_responses[emo]:
            all_ref_texts.append(resp)
            all_ref_labels.append(emo)

    if not all_ref_texts or not any(len(t.strip()) > 0 for t in all_ref_texts):
        return {"error": "No valid reference responses generated"}

    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    try:
        ref_vecs = tfidf.fit_transform(all_ref_texts).toarray()
    except ValueError:
        return {"error": "TF-IDF failed (empty vocabulary)"}

    # Compute per-emotion reference centroid
    ref_centroids = {}
    for emo in EMOTIONS_5:
        idx = [i for i, l in enumerate(all_ref_labels) if l == emo]
        if idx:
            ref_centroids[emo] = ref_vecs[idx].mean(axis=0)

    # For each conflict response, check similarity to label vs context emotion
    behavioral_results = []
    label_beh_wins = 0
    ctx_beh_wins = 0

    for ci, s in enumerate(conflict_items):
        resp = conflict_responses[ci]
        try:
            resp_vec = tfidf.transform([resp]).toarray()[0]
        except Exception:
            behavioral_results.append({
                "item_id": s["item_id"],
                "label_emotion": s["emotion"],
                "context_emotion": s["conflict_context_emotion"],
                "winner": "unknown",
                "response_preview": resp[:200],
            })
            continue

        label_emo = s["emotion"]
        ctx_emo = s["conflict_context_emotion"]

        if label_emo in ref_centroids and ctx_emo in ref_centroids:
            sim_label = float(sk_cosine([resp_vec], [ref_centroids[label_emo]])[0, 0])
            sim_ctx = float(sk_cosine([resp_vec], [ref_centroids[ctx_emo]])[0, 0])
            winner = "label" if sim_label > sim_ctx else "context"
            if winner == "label":
                label_beh_wins += 1
            else:
                ctx_beh_wins += 1
        else:
            sim_label = float("nan")
            sim_ctx = float("nan")
            winner = "unknown"

        behavioral_results.append({
            "item_id": s["item_id"],
            "label_emotion": label_emo,
            "context_emotion": ctx_emo,
            "sim_to_label_responses": round(sim_label, 4),
            "sim_to_context_responses": round(sim_ctx, 4),
            "winner": winner,
            "response_preview": resp[:200],
        })

    total_beh = label_beh_wins + ctx_beh_wins
    return {
        "label_behavioral_rate": round(label_beh_wins / max(total_beh, 1), 3),
        "context_behavioral_rate": round(ctx_beh_wins / max(total_beh, 1), 3),
        "label_wins": label_beh_wins,
        "context_wins": ctx_beh_wins,
        "total": total_beh,
        "details": behavioral_results,
    }


def _generate_one(model, tokenizer, text, device, max_new=150):
    """Generate a single response."""
    messages = [{"role": "user", "content": text}]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_text = f"User: {text}\nAssistant:"

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
    return resp


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Label-Context Conflict RSA: extraction + causal analysis")
    ap.add_argument("--model_path", required=True, help="Path to HF model")
    ap.add_argument("--model_short", default=None, help="Short model name")
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peak = MODELS_PEAK.get(model_short, 15)

    print(f"{'='*70}")
    print(f"LABEL-CONTEXT CONFLICT RSA: {model_short}")
    print(f"{'='*70}")
    print(f"Device: {device}, Peak layer: L{peak}")

    t0 = time.time()

    # ── Load stimuli ──
    with open(STIM_PATH) as f:
        stimuli = [json.loads(line) for line in f]
    print(f"Loaded {len(stimuli)} stimuli from {STIM_PATH.name}")

    n_label = sum(1 for s in stimuli if s["condition"] == "label_only")
    n_ctx = sum(1 for s in stimuli if s["condition"] == "context_only")
    n_conf = sum(1 for s in stimuli if s["condition"] == "conflict")
    print(f"  label_only={n_label}, context_only={n_ctx}, conflict={n_conf}")

    # ── Load brain within-affective RDM ──
    brain_rdm_5x5, brain_emo_order = load_brain_affective_rdm()
    print(f"Brain affective RDM: {brain_rdm_5x5.shape} ({brain_emo_order})")

    # ── Load model ──
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
    print(f"  {n_layers} layers, peak L{peak}")

    # ═══ PHASE 1: Extract hidden states ═══
    print(f"\n{'='*70}")
    print("PHASE 1: Extract hidden states")
    print(f"{'='*70}")
    t1 = time.time()
    peak_states, all_layer_states = extract_hidden_states(
        model, tokenizer, stimuli, device, n_layers, peak)
    print(f"  Extraction done ({time.time()-t1:.0f}s)")
    print(f"  peak_states: {peak_states.shape}")
    print(f"  all_layer_states: {all_layer_states.shape}")

    # ═══ PHASE 2: Compute label_only centroids ═══
    print(f"\n{'='*70}")
    print("PHASE 2: Compute emotion centroids from label_only")
    print(f"{'='*70}")
    centroids = compute_centroids(peak_states, stimuli, "label_only")
    for emo in EMOTIONS_5:
        print(f"  {emo:12s}: norm={np.linalg.norm(centroids[emo]):.4f}")

    # ═══ PHASE 3: Context-only accuracy ═══
    print(f"\n{'='*70}")
    print("PHASE 3: Context-only centroid proximity (emotion identification)")
    print(f"{'='*70}")
    ctx_results = analyze_context_only(peak_states, stimuli, centroids)
    print(f"  Overall accuracy: {ctx_results['accuracy']:.3f} "
          f"({ctx_results['correct']}/{ctx_results['total']})")
    for emo in EMOTIONS_5:
        pe = ctx_results["per_emotion"][emo]
        print(f"    {emo:12s}: {pe['accuracy']:.2f} "
              f"({pe['correct']}/{pe['total']})")
    if ctx_results["confusions"]:
        print(f"  Confusions ({len(ctx_results['confusions'])}):")
        for c in ctx_results["confusions"][:5]:
            print(f"    item {c['item_id']}: true={c['true_emotion']}, "
                  f"pred={c['predicted']}")

    # ═══ PHASE 4: Conflict analysis (THE KEY TEST) ═══
    print(f"\n{'='*70}")
    print("PHASE 4: CONFLICT ANALYSIS (label vs context)")
    print(f"{'='*70}")
    conflict_results = analyze_conflict(peak_states, stimuli, centroids)
    print(f"  Label wins:   {conflict_results['label_wins']}/{conflict_results['total']} "
          f"({conflict_results['label_rate']:.1%})")
    print(f"  Context wins: {conflict_results['context_wins']}/{conflict_results['total']} "
          f"({conflict_results['context_rate']:.1%})")
    print(f"\n  Per-emotion breakdown:")
    for emo in EMOTIONS_5:
        pe = conflict_results["per_emotion"][emo]
        print(f"    {emo:12s} (label) vs {pe['context_emotion']:12s} (context): "
              f"label={pe['label_rate']:.0%}, context={pe['context_rate']:.0%}")

    interpretation = ""
    if conflict_results["label_rate"] > 0.7:
        interpretation = "STRONGLY LABEL-DRIVEN: model representations dominated by explicit emotion labels"
    elif conflict_results["label_rate"] > 0.5:
        interpretation = "LABEL-LEANING: model representations somewhat driven by labels over context"
    elif conflict_results["context_rate"] > 0.7:
        interpretation = "STRONGLY CONTEXT-DRIVEN: model performs appraisal-based inference from context"
    elif conflict_results["context_rate"] > 0.5:
        interpretation = "CONTEXT-LEANING: model partially infers emotion from situational context"
    else:
        interpretation = "MIXED: no clear dominance of label or context"
    print(f"\n  INTERPRETATION: {interpretation}")

    # ═══ PHASE 5: Condition-level RDMs + brain comparison ═══
    print(f"\n{'='*70}")
    print("PHASE 5: Condition-level 5x5 RDMs")
    print(f"{'='*70}")

    label_rdm = build_condition_rdm(peak_states, stimuli, "label_only")
    ctx_rdm = build_condition_rdm(peak_states, stimuli, "context_only")

    label_rho, label_p = compare_to_brain(label_rdm, brain_rdm_5x5)
    ctx_rho, ctx_p = compare_to_brain(ctx_rdm, brain_rdm_5x5)

    print(f"  label_only  RDM vs brain: rho={label_rho:+.4f} (p={label_p:.4f})")
    print(f"  context_only RDM vs brain: rho={ctx_rho:+.4f} (p={ctx_p:.4f})")
    print(f"  Delta (context - label): {ctx_rho - label_rho:+.4f}")

    # Print RDMs
    print(f"\n  label_only 5x5 RDM:")
    for i, emo in enumerate(EMOTIONS_5):
        vals = " ".join(f"{label_rdm[i, j]:.4f}" for j in range(5))
        print(f"    {emo:12s}: {vals}")

    print(f"\n  context_only 5x5 RDM:")
    for i, emo in enumerate(EMOTIONS_5):
        vals = " ".join(f"{ctx_rdm[i, j]:.4f}" for j in range(5))
        print(f"    {emo:12s}: {vals}")

    # Mean pairwise distance (higher = better differentiation)
    label_mean_d = float(triu_vals(label_rdm).mean())
    ctx_mean_d = float(triu_vals(ctx_rdm).mean())
    print(f"\n  Mean pairwise distance: label_only={label_mean_d:.4f}, "
          f"context_only={ctx_mean_d:.4f}")
    print(f"  Context/Label ratio: {ctx_mean_d / max(label_mean_d, 1e-10):.3f}")

    # ═══ PHASE 5b: Layer sweep for conflict ═══
    print(f"\n{'='*70}")
    print("PHASE 5b: Layer sweep (conflict label_rate per layer)")
    print(f"{'='*70}")
    layer_sweep = layer_sweep_conflict(all_layer_states, stimuli, n_layers)
    peak_label_layer = int(np.argmax(layer_sweep))
    print(f"  Layer with highest label_rate: L{peak_label_layer} "
          f"({layer_sweep[peak_label_layer]:.1%})")
    print(f"  Label rate at peak L{peak}: {layer_sweep[peak]:.1%}")
    # Print every 5th layer
    for L in range(0, len(layer_sweep), max(1, n_layers // 8)):
        print(f"    L{L:2d}: {layer_sweep[L]:.1%}")

    # ═══ PHASE 6: Behavioral test ═══
    print(f"\n{'='*70}")
    print("PHASE 6: Behavioral conflict test (response generation)")
    print(f"{'='*70}")
    behavioral = generate_and_analyze_behavioral(
        model, tokenizer, stimuli, peak_states, centroids, device)

    if "error" not in behavioral:
        print(f"  Behavioral label rate:   {behavioral['label_behavioral_rate']:.1%}")
        print(f"  Behavioral context rate: {behavioral['context_behavioral_rate']:.1%}")
        print(f"\n  Sample conflict responses:")
        for d in behavioral["details"][:3]:
            print(f"    item {d['item_id']} ({d['label_emotion']} label, "
                  f"{d['context_emotion']} context):")
            print(f"      -> {d.get('response_preview', 'N/A')[:120]}...")
            print(f"      winner: {d['winner']}")
    else:
        print(f"  Behavioral test error: {behavioral['error']}")

    # ═══ SUMMARY ═══
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_short}")
    print(f"{'='*70}")
    print(f"  Context-only accuracy:        {ctx_results['accuracy']:.3f}")
    print(f"  Conflict label rate:          {conflict_results['label_rate']:.3f}")
    print(f"  Conflict context rate:        {conflict_results['context_rate']:.3f}")
    print(f"  label_only RDM vs brain:      rho={label_rho:+.4f}")
    print(f"  context_only RDM vs brain:    rho={ctx_rho:+.4f}")
    print(f"  Mean dist (label_only):       {label_mean_d:.4f}")
    print(f"  Mean dist (context_only):     {ctx_mean_d:.4f}")
    if "error" not in behavioral:
        print(f"  Behavioral label rate:        {behavioral['label_behavioral_rate']:.3f}")
    print(f"  INTERPRETATION: {interpretation}")

    # ═══ SAVE ═══
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up details for JSON serialization
    conflict_details_clean = []
    for d in conflict_results["details"]:
        conflict_details_clean.append({
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in d.items()
        })

    output = {
        "model": model_short,
        "peak_layer": peak,
        "n_stimuli": len(stimuli),
        "n_layers": n_layers,
        "context_only_accuracy": round(ctx_results["accuracy"], 4),
        "context_only_per_emotion": {
            emo: round(ctx_results["per_emotion"][emo]["accuracy"], 4)
            for emo in EMOTIONS_5
        },
        "conflict_label_rate": conflict_results["label_rate"],
        "conflict_context_rate": conflict_results["context_rate"],
        "per_emotion_conflict": conflict_results["per_emotion"],
        "label_only_rdm_5x5": label_rdm.tolist(),
        "context_only_rdm_5x5": ctx_rdm.tolist(),
        "brain_comparison": {
            "label_only_rho_vs_brain": round(label_rho, 4),
            "label_only_p_vs_brain": round(label_p, 4),
            "context_only_rho_vs_brain": round(ctx_rho, 4),
            "context_only_p_vs_brain": round(ctx_p, 4),
            "label_only_mean_dist": round(label_mean_d, 5),
            "context_only_mean_dist": round(ctx_mean_d, 5),
        },
        "behavioral_conflict": {
            k: v for k, v in behavioral.items() if k != "details"
        } if "error" not in behavioral else {"error": behavioral["error"]},
        "behavioral_details": behavioral.get("details", []),
        "layer_sweep": layer_sweep,
        "conflict_details": conflict_details_clean,
        "interpretation": interpretation,
        "emotions": EMOTIONS_5,
        "conflict_pairings": CONFLICT_MAP,
    }

    out_path = OUT_DIR / f"{model_short}_label_context_conflict.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
