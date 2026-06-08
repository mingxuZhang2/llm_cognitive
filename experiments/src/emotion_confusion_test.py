#!/usr/bin/env python3
"""
Emotion confusion test: does LLM emotion geometry predict response confusion?

If anger and sadness are close in LLM space but far in brain space, the model
should give similar responses to "I feel angry about X" and "I feel sad about X"
even though humans would respond very differently.

Method:
  1. Create matched scenario pairs: same situation, different emotions
  2. Get model's response to each
  3. Measure response similarity (hidden state cosine at last layer)
  4. Correlate with LLM emotion RDM distances
  5. Show that LLM-close pairs get more similar responses

GPU needed.

Output:
  results/mechanistic/emotion_confusion_test.json
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
OUT_DIR.mkdir(exist_ok=True)

AFF = ["anger", "fear", "disgust", "sadness", "happiness"]
# Exclude valence — it's not a discrete emotion

SCENARIOS = [
    "My best friend lied to me about something important.",
    "I found out my coworker has been taking credit for my work.",
    "Someone cut in line right in front of me after I'd been waiting for an hour.",
    "I discovered that a company has been polluting the river near my hometown.",
    "My partner forgot our anniversary even though I reminded them twice.",
    "I watched a documentary about animal cruelty in factory farms.",
    "A stranger on the internet attacked me personally for sharing my opinion.",
    "I learned that my childhood home is being demolished.",
    "My doctor told me my test results were concerning and I need more tests.",
    "I saw someone being incredibly rude to an elderly person on the bus.",
]

EMOTION_PROMPTS = {
    "anger": "This makes me feel angry.",
    "fear": "This makes me feel afraid.",
    "disgust": "This makes me feel disgusted.",
    "sadness": "This makes me feel sad.",
    "happiness": "Despite everything, this situation makes me feel strangely at peace.",
}


def get_response_embedding(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    last_hidden = out.hidden_states[-1][0]
    mask = inputs["attention_mask"][0].bool()
    mean_emb = last_hidden[mask].mean(dim=0).cpu().numpy()
    return mean_emb


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Emotion Confusion Test ({model_short}) ===")
    print(f"Device: {device}")
    print(f"{len(SCENARIOS)} scenarios x {len(AFF)} emotions = {len(SCENARIOS)*len(AFF)} prompts")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Step 1: Get response embeddings for each scenario x emotion
    print("\n--- Step 1: Computing response embeddings ---")
    embeddings = {}  # (scenario_idx, emotion) -> embedding

    for si, scenario in enumerate(SCENARIOS):
        for emo in AFF:
            prompt = f"{scenario} {EMOTION_PROMPTS[emo]} How should I deal with this?"
            emb = get_response_embedding(model, tokenizer, prompt, device)
            embeddings[(si, emo)] = emb
        if (si + 1) % 5 == 0:
            print(f"  [{si+1}/{len(SCENARIOS)}]")

    # Step 2: For each emotion pair, compute mean response similarity across scenarios
    print("\n--- Step 2: Response similarity per emotion pair ---")
    pair_similarities = {}

    for e1, e2 in itertools.combinations(AFF, 2):
        sims = []
        for si in range(len(SCENARIOS)):
            emb1 = embeddings[(si, e1)]
            emb2 = embeddings[(si, e2)]
            sim = 1 - cosine(emb1, emb2)
            sims.append(sim)
        pair_similarities[(e1, e2)] = {
            "mean_cosine": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "per_scenario": [float(s) for s in sims],
        }

    # Step 3: Load LLM and brain within-affective RDMs
    print("\n--- Step 3: Correlating with geometry ---")

    # Find the model's RDM file
    model_files = {
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
        "Qwen": "Qwen2.5-7B-Instruct",
    }
    rdm_name = model_files.get(model_short, model_short)
    rdm_path = RSA / f"{rdm_name}_rdm14_headline.npz"

    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_conds = list(ns["conditions"])
    brain_rdm = ns["rdm"]

    if rdm_path.exists():
        z = np.load(rdm_path, allow_pickle=True)
        llm_conds = list(z["conditions"])
        llm_rdm = z["rdm"]
    else:
        print(f"  WARNING: {rdm_path} not found, using avg of all models")
        llm_rdm = None

    # Build vectors for correlation
    pair_keys = list(itertools.combinations(AFF, 2))
    response_sim_vec = np.array([pair_similarities[k]["mean_cosine"] for k in pair_keys])

    brain_dist_vec = np.array([
        brain_rdm[brain_conds.index(e1), brain_conds.index(e2)]
        for e1, e2 in pair_keys
    ])

    if llm_rdm is not None:
        llm_dist_vec = np.array([
            llm_rdm[llm_conds.index(e1), llm_conds.index(e2)]
            for e1, e2 in pair_keys
        ])
    else:
        llm_dist_vec = None

    # Response similarity should NEGATIVELY correlate with distance
    # (close pairs → similar responses → high cosine)
    rho_brain, p_brain = spearmanr(response_sim_vec, brain_dist_vec)
    print(f"\n  Response similarity vs brain distance: rho={rho_brain:+.3f} (p={p_brain:.4f})")

    if llm_dist_vec is not None:
        rho_llm, p_llm = spearmanr(response_sim_vec, llm_dist_vec)
        print(f"  Response similarity vs LLM distance:   rho={rho_llm:+.3f} (p={p_llm:.4f})")
    else:
        rho_llm, p_llm = None, None

    # Step 4: Print pair-by-pair results
    print(f"\n--- Pair-by-pair results ---")
    print(f"{'Pair':>25s} | {'Resp sim':>8s} | {'Brain dist':>10s} | {'LLM dist':>10s} | {'Prediction':>10s}")
    print("-" * 75)

    for e1, e2 in sorted(pair_keys, key=lambda k: -pair_similarities[k]["mean_cosine"]):
        rs = pair_similarities[(e1, e2)]["mean_cosine"]
        bd = brain_rdm[brain_conds.index(e1), brain_conds.index(e2)]
        ld = llm_rdm[llm_conds.index(e1), llm_conds.index(e2)] if llm_rdm is not None else 0

        # Flag confusion cases: high response sim + high brain distance
        if rs > np.median(response_sim_vec) and bd > np.median(brain_dist_vec):
            flag = "CONFUSE"
        else:
            flag = ""
        print(f"{e1+'-'+e2:>25s} | {rs:8.4f} | {bd:10.4f} | {ld:10.4f} | {flag:>10s}")

    # Save results
    results = {
        "model": model_short,
        "n_scenarios": len(SCENARIOS),
        "n_emotions": len(AFF),
        "pair_response_similarity": {
            f"{e1}-{e2}": pair_similarities[(e1, e2)]
            for e1, e2 in pair_keys
        },
        "correlation_with_brain_distance": {
            "rho": float(rho_brain), "p": float(p_brain),
        },
        "correlation_with_llm_distance": {
            "rho": float(rho_llm) if rho_llm is not None else None,
            "p": float(p_llm) if p_llm is not None else None,
        },
        "scenarios": SCENARIOS,
    }

    out_path = OUT_DIR / f"emotion_confusion_test_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
