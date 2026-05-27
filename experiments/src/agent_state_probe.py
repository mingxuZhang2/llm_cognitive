#!/usr/bin/env python3
"""
Experiment B: Agent-state factorization probe.

Test whether LLMs factorize mental state variables (belief, desire, intention,
knowledge) into separable representational dimensions, or collapse them.

Method:
1. Create synthetic stories with controlled latent variables:
   - Same surface form, different mental state type
   - e.g., "Alice thinks X" (belief) vs "Alice wants X" (desire)
         vs "Alice plans to X" (intention) vs "Alice knows X" (knowledge)
2. Extract hidden states for each story
3. Train linear probes to decode each latent variable
4. Measure: can the probe distinguish belief from desire from intention?
5. Compare probe accuracy with brain RDM distances for same conditions

If probes fail specifically on pairs that the brain distinguishes but the LLM
RDM collapses → confirms the factorization deficit.

Usage:
  python agent_state_probe.py \
    --model_path /path/to/model \
    --model_short Qwen2.5-3B-Instruct \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

# Synthetic stimuli: controlled minimal pairs
# Each story has the same surface structure but varies ONE mental state type
AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
OBJECTS = [
    "the meeting is at 3pm", "the package has arrived", "the store is closed",
    "the test is tomorrow", "the flight is delayed", "the project is cancelled",
    "the key is under the mat", "the password has changed",
    "the deadline has moved", "the restaurant is fully booked",
]

MENTAL_STATE_TEMPLATES = {
    "belief": [
        "{agent} believes that {object}.",
        "{agent} is convinced that {object}.",
        "{agent} thinks that {object}.",
        "{agent} assumes that {object}.",
    ],
    "desire": [
        "{agent} wants {object_desire}.",
        "{agent} hopes for {object_desire}.",
        "{agent} wishes that {object_desire}.",
        "{agent} longs for {object_desire}.",
    ],
    "intention": [
        "{agent} plans to {object_action}.",
        "{agent} intends to {object_action}.",
        "{agent} has decided to {object_action}.",
        "{agent} is going to {object_action}.",
    ],
    "knowledge": [
        "{agent} knows that {object}.",
        "{agent} is aware that {object}.",
        "{agent} has learned that {object}.",
        "{agent} found out that {object}.",
    ],
    "ignorance": [
        "{agent} doesn't know that {object}.",
        "{agent} is unaware that {object}.",
        "{agent} hasn't learned that {object}.",
        "{agent} has no idea that {object}.",
    ],
}

DESIRE_OBJECTS = [
    "a promotion", "a vacation", "a new laptop", "more free time",
    "better grades", "a quiet evening", "recognition from peers",
    "a fresh start", "more responsibility", "a creative outlet",
]

ACTION_OBJECTS = [
    "finish the report", "call the client", "book the flight",
    "start the experiment", "resign from the position", "apply for the grant",
    "confront the colleague", "reorganize the team", "cancel the subscription",
    "move to a new city",
]


def generate_stimuli(n_per_condition=80):
    """Generate controlled synthetic stimuli."""
    import random
    rng = random.Random(42)
    stimuli = []

    for ms_type, templates in MENTAL_STATE_TEMPLATES.items():
        for _ in range(n_per_condition):
            agent = rng.choice(AGENTS)
            template = rng.choice(templates)

            if ms_type == "desire":
                obj = rng.choice(DESIRE_OBJECTS)
                text = template.format(agent=agent, object_desire=obj)
            elif ms_type == "intention":
                obj = rng.choice(ACTION_OBJECTS)
                text = template.format(agent=agent, object_action=obj)
            else:
                obj = rng.choice(OBJECTS)
                text = template.format(agent=agent, object=obj)

            stimuli.append({"text": text, "mental_state": ms_type, "agent": agent})

    rng.shuffle(stimuli)
    return stimuli


def extract_activations(model, tokenizer, texts, device, batch_size=16):
    """Extract last-layer mean-pooled activations."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last_hs = out.hidden_states[-1]
        for b in range(len(batch)):
            mask = inputs["attention_mask"][b].bool()
            mean_act = last_hs[b][mask].mean(dim=0).cpu().numpy()
            all_acts.append(mean_act)
    return np.array(all_acts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_per_condition", type=int, default=80)
    args = parser.parse_args()

    # Generate stimuli
    stimuli = generate_stimuli(args.n_per_condition)
    texts = [s["text"] for s in stimuli]
    labels = [s["mental_state"] for s in stimuli]
    ms_types = sorted(set(labels))
    print(f"Generated {len(stimuli)} stimuli across {len(ms_types)} mental states")
    for ms in ms_types:
        print(f"  {ms}: {sum(1 for l in labels if l == ms)}")

    # Load model
    print(f"\nLoading {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # Extract activations
    print("Extracting activations...")
    acts = extract_activations(model, tokenizer, texts, device)
    print(f"  Shape: {acts.shape}")

    # Build RDM from mental state centroids
    centroids = {}
    for ms in ms_types:
        idx = [i for i, l in enumerate(labels) if l == ms]
        centroids[ms] = acts[idx].mean(axis=0)

    centroid_mat = np.array([centroids[ms] for ms in ms_types])
    centroid_mat = centroid_mat - centroid_mat.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centroid_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    rdm = 1.0 - np.clip((centroid_mat / norms) @ (centroid_mat / norms).T, -1, 1)

    print(f"\nMental State RDM ({len(ms_types)}×{len(ms_types)}):")
    print(f"{'':>12s}" + "".join(f"{ms[:8]:>10s}" for ms in ms_types))
    for i, ms_i in enumerate(ms_types):
        row = "".join(f"{rdm[i,j]:>10.4f}" for j in range(len(ms_types)))
        print(f"{ms_i:>12s}{row}")

    # Pairwise probe: can we decode which mental state type from activations?
    print(f"\n=== PAIRWISE LINEAR PROBE (5-fold CV) ===")
    label_array = np.array(labels)
    pair_results = []

    for i, ms_a in enumerate(ms_types):
        for j, ms_b in enumerate(ms_types):
            if j <= i:
                continue

            idx_a = np.where(label_array == ms_a)[0]
            idx_b = np.where(label_array == ms_b)[0]
            X = np.vstack([acts[idx_a], acts[idx_b]])
            y = np.array([0] * len(idx_a) + [1] * len(idx_b))

            clf = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            acc = scores.mean()

            pair_results.append({
                "ms_a": ms_a, "ms_b": ms_b,
                "probe_accuracy": float(acc),
                "rdm_distance": float(rdm[i, j]),
            })
            print(f"  {ms_a:>12s} vs {ms_b:<12s}  probe acc={acc:.1%}  RDM dist={rdm[i,j]:.4f}")

    # Overall probe: 5-way classification
    print(f"\n=== 5-WAY CLASSIFICATION PROBE ===")
    y_multi = np.array([ms_types.index(l) for l in labels])
    clf_multi = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
    scores_multi = cross_val_score(clf_multi, acts, y_multi, cv=5, scoring="accuracy")
    print(f"  5-way probe accuracy: {scores_multi.mean():.1%} ± {scores_multi.std():.1%}")
    print(f"  (Chance = {1/len(ms_types):.1%})")

    # Key question: which pairs are hardest to probe?
    sorted_pairs = sorted(pair_results, key=lambda x: x["probe_accuracy"])
    print(f"\n  3 HARDEST to distinguish:")
    for p in sorted_pairs[:3]:
        print(f"    {p['ms_a']:>12s} vs {p['ms_b']:<12s}  acc={p['probe_accuracy']:.1%}")
    print(f"\n  3 EASIEST to distinguish:")
    for p in sorted_pairs[-3:]:
        print(f"    {p['ms_a']:>12s} vs {p['ms_b']:<12s}  acc={p['probe_accuracy']:.1%}")

    # Does RDM distance predict probe accuracy?
    rdm_dists = [p["rdm_distance"] for p in pair_results]
    probe_accs = [p["probe_accuracy"] for p in pair_results]
    rho_rp, p_rp = spearmanr(rdm_dists, probe_accs)
    print(f"\n  RDM distance vs probe accuracy: ρ = {rho_rp:+.4f}, p = {p_rp:.4f}")

    # Save
    output = {
        "model": args.model_short,
        "n_per_condition": args.n_per_condition,
        "mental_states": ms_types,
        "rdm": rdm.tolist(),
        "five_way_accuracy": float(scores_multi.mean()),
        "five_way_std": float(scores_multi.std()),
        "pairwise_probes": pair_results,
        "rdm_vs_probe_corr": {"rho": float(rho_rp), "p": float(p_rp)},
    }
    out_path = Path(args.output_dir) / f"{args.model_short}_agent_state_probe.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
