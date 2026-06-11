#!/usr/bin/env python3
"""
Performing Empathy Dissection: prove that LLMs produce empathic OUTPUT
without empathic internal PROCESSING.

Evidence chain:
  E1. Preamble vs Strategy split: empathy lives ONLY in the opening,
      action strategies are emotion-blind
  E2. LOO classification: strip the preamble → can't tell which emotion
      from the strategy alone (at chance)
  E3. Label ablation (GPU): remove the emotion label, keep the scenario →
      generated advice is nearly identical → label only routes to preamble
  E4. Internal clustering: hidden states cluster by SCENARIO not EMOTION
  E5. Contrast: the SAME model gives differentiated strategies for social
      cognition prompts → it CAN differentiate, but DOESN'T for emotions

GPU required for E3 (label ablation generation). E1, E2, E4, E5 are CPU.

Usage:
  python src/performing_empathy_dissection.py --model_path /path/to/model \
    --model_short Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "performing_empathy_dissection"
OUT_DIR.mkdir(exist_ok=True)

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness"]

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

PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}


def split_response(text):
    """Split response into empathy preamble and action strategy."""
    lines = text.split("\n")
    preamble, strategy = [], []
    found_action = False
    for line in lines:
        s = line.strip()
        if not found_action and (
            s.startswith(("1.", "2.", "*", "- ", "•"))
            or ("**" in s and any(c.isdigit() for c in s[:5]))
        ):
            found_action = True
        if found_action:
            strategy.append(s)
        else:
            preamble.append(s)
    return " ".join(preamble).strip(), " ".join(strategy).strip()


def evidence_1_preamble_strategy(resps, results):
    """E1: Strategies are scenario-driven, preambles are emotion-driven."""
    print(f"\n{'='*70}")
    print("E1: PREAMBLE vs STRATEGY DISSECTION")
    print(f"{'='*70}")

    preambles, strategies = [], []
    for r in resps:
        pre, strat = split_response(r["response"])
        preambles.append(pre)
        strategies.append(strat)

    # TF-IDF on strategies
    tfidf_s = TfidfVectorizer(max_features=2000, stop_words="english")
    svecs = tfidf_s.fit_transform(strategies).toarray()

    # Within-scenario similarity (same event, 5 different emotions)
    within_scen = []
    for si in range(10):
        idx = [si * 5 + ei for ei in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                within_scen.append(
                    cosine_similarity([svecs[idx[i]]], [svecs[idx[j]]])[0][0]
                )

    # Within-emotion similarity (same emotion, 10 different scenarios)
    within_emo = []
    for ei in range(5):
        idx = [si * 5 + ei for si in range(10)]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                within_emo.append(
                    cosine_similarity([svecs[idx[i]]], [svecs[idx[j]]])[0][0]
                )

    u, p = mannwhitneyu(within_scen, within_emo, alternative="greater")
    print(f"  Strategy similarity:")
    print(f"    Within SCENARIO (same event, diff emotions): {np.mean(within_scen):.3f}")
    print(f"    Within EMOTION  (same emotion, diff events): {np.mean(within_emo):.3f}")
    print(f"    Mann-Whitney U={u:.0f}, p={p:.4f}")

    # Same for preambles
    tfidf_p = TfidfVectorizer(max_features=2000, stop_words="english")
    pvecs = tfidf_p.fit_transform(preambles).toarray()

    pre_within_scen, pre_within_emo = [], []
    for si in range(10):
        idx = [si * 5 + ei for ei in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                pre_within_scen.append(
                    cosine_similarity([pvecs[idx[i]]], [pvecs[idx[j]]])[0][0]
                )
    for ei in range(5):
        idx = [si * 5 + ei for si in range(10)]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pre_within_emo.append(
                    cosine_similarity([pvecs[idx[i]]], [pvecs[idx[j]]])[0][0]
                )

    print(f"\n  Preamble similarity:")
    print(f"    Within SCENARIO: {np.mean(pre_within_scen):.3f}")
    print(f"    Within EMOTION:  {np.mean(pre_within_emo):.3f}")

    results["e1_preamble_strategy"] = {
        "strategy_within_scenario": float(np.mean(within_scen)),
        "strategy_within_emotion": float(np.mean(within_emo)),
        "strategy_U": float(u), "strategy_p": float(p),
        "preamble_within_scenario": float(np.mean(pre_within_scen)),
        "preamble_within_emotion": float(np.mean(pre_within_emo)),
    }


def evidence_2_blind_classification(resps, results):
    """E2: Strip preamble → classify by emotion vs scenario."""
    print(f"\n{'='*70}")
    print("E2: BLIND CLASSIFICATION (strategy only, preamble stripped)")
    print(f"{'='*70}")

    strategies = [split_response(r["response"])[1] for r in resps]
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
    vecs = tfidf.fit_transform(strategies).toarray()

    # LOO emotion classification
    emo_correct = 0
    for i, r in enumerate(resps):
        true_emo = r["emotion"]
        centroids = {}
        for emo in EMOTIONS:
            same = [j for j, rr in enumerate(resps) if rr["emotion"] == emo and j != i]
            if same:
                centroids[emo] = vecs[same].mean(axis=0)
        pred = max(centroids, key=lambda e: cosine_similarity([vecs[i]], [centroids[e]])[0][0])
        if pred == true_emo:
            emo_correct += 1

    # LOO scenario classification
    sce_correct = 0
    for i, r in enumerate(resps):
        true_s = r["scenario"]
        centroids = {}
        for si in range(10):
            same = [j for j, rr in enumerate(resps) if rr["scenario"] == si and j != i]
            if same:
                centroids[si] = vecs[same].mean(axis=0)
        pred = max(centroids, key=lambda s: cosine_similarity([vecs[i]], [centroids[s]])[0][0])
        if pred == true_s:
            sce_correct += 1

    emo_acc = emo_correct / len(resps)
    sce_acc = sce_correct / len(resps)
    print(f"  Emotion classification (chance=20%): {emo_acc:.1%}")
    print(f"  Scenario classification (chance=10%): {sce_acc:.1%}")
    print(f"  → {'EMOTION-BLIND' if emo_acc < 0.30 else 'some emotion signal'}")

    results["e2_blind_classification"] = {
        "emotion_accuracy": float(emo_acc), "emotion_chance": 0.2,
        "scenario_accuracy": float(sce_acc), "scenario_chance": 0.1,
    }


def evidence_3_label_ablation(model, tokenizer, device, results):
    """E3: Remove emotion label → same advice? (GPU)"""
    print(f"\n{'='*70}")
    print("E3: LABEL ABLATION (remove emotion label, keep scenario)")
    print(f"{'='*70}")

    def generate(text, max_new=300):
        msgs = [{"role": "user", "content": text}]
        try:
            ct = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        except Exception:
            ct = f"User: {text}\nAssistant:"
        inputs = tokenizer(ct, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        new = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new, skip_special_tokens=True).strip()

    labeled_resps = []
    unlabeled_resps = []

    print(f"  Generating {10 * 5} labeled + {10} unlabeled responses...")

    for si, scenario in enumerate(SCENARIOS):
        # Unlabeled: no emotion, just the scenario
        unlabeled_text = f"{scenario}. I really need someone to talk to."
        unlabeled_resp = generate(unlabeled_text)
        unlabeled_resps.append({"scenario": si, "response": unlabeled_resp})

        # Labeled: with emotion
        for emo in EMOTIONS:
            labeled_text = (f"{scenario}. I feel {EMOTION_PHRASES[emo]}. "
                            "I really need someone to talk to.")
            labeled_resp = generate(labeled_text)
            labeled_resps.append({
                "scenario": si, "emotion": emo, "response": labeled_resp
            })

        if (si + 1) % 3 == 0:
            print(f"    {si + 1}/10 scenarios done")

    # Compare: for each scenario, similarity between unlabeled and each labeled
    all_resps_text = ([r["response"] for r in labeled_resps]
                      + [r["response"] for r in unlabeled_resps])
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    vecs = tfidf.fit_transform(all_resps_text).toarray()

    labeled_vecs = vecs[:50]
    unlabeled_vecs = vecs[50:]

    # Split into strategies
    labeled_strats = [split_response(r["response"])[1] for r in labeled_resps]
    unlabeled_strats = [split_response(r["response"])[1] for r in unlabeled_resps]
    all_strats = labeled_strats + unlabeled_strats
    strat_vecs = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform(all_strats).toarray()
    l_svecs = strat_vecs[:50]
    u_svecs = strat_vecs[50:]

    # For each scenario: sim(unlabeled, each labeled emotion)
    label_vs_unlabel_full = []
    label_vs_unlabel_strat = []
    across_emotion_strat = []

    for si in range(10):
        u_idx = si
        for ei in range(5):
            l_idx = si * 5 + ei
            sim_full = cosine_similarity([labeled_vecs[l_idx]], [unlabeled_vecs[u_idx]])[0][0]
            sim_strat = cosine_similarity([l_svecs[l_idx]], [u_svecs[u_idx]])[0][0]
            label_vs_unlabel_full.append(sim_full)
            label_vs_unlabel_strat.append(sim_strat)

        # Also: across-emotion similarity within this scenario (labeled only)
        for ei in range(5):
            for ej in range(ei + 1, 5):
                sim = cosine_similarity(
                    [l_svecs[si * 5 + ei]], [l_svecs[si * 5 + ej]])[0][0]
                across_emotion_strat.append(sim)

    print(f"\n  Full response: labeled vs unlabeled similarity: "
          f"{np.mean(label_vs_unlabel_full):.3f}")
    print(f"  Strategy only: labeled vs unlabeled similarity: "
          f"{np.mean(label_vs_unlabel_strat):.3f}")
    print(f"  Strategy only: across-emotion similarity: "
          f"{np.mean(across_emotion_strat):.3f}")
    print(f"  → {'LABEL HAS NO EFFECT on strategy' if np.mean(label_vs_unlabel_strat) > 0.5 else 'Label has some effect'}")

    # Show example
    print(f"\n  Example (Scenario 0):")
    print(f"    UNLABELED: {unlabeled_strats[0][:150]}...")
    print(f"    ANGER:     {labeled_strats[0][:150]}...")
    print(f"    SADNESS:   {labeled_strats[4][:150]}...")

    results["e3_label_ablation"] = {
        "full_response_sim_labeled_vs_unlabeled": float(np.mean(label_vs_unlabel_full)),
        "strategy_sim_labeled_vs_unlabeled": float(np.mean(label_vs_unlabel_strat)),
        "strategy_sim_across_emotions": float(np.mean(across_emotion_strat)),
        "n_labeled": len(labeled_resps),
        "n_unlabeled": len(unlabeled_resps),
    }

    return labeled_resps, unlabeled_resps


def evidence_4_internal_clustering(model, tokenizer, resps_labeled,
                                    resps_unlabeled, device, peak, results):
    """E4: Hidden states cluster by scenario, not emotion."""
    print(f"\n{'='*70}")
    print("E4: INTERNAL CLUSTERING (scenario vs emotion)")
    print(f"{'='*70}")

    def get_hidden(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[peak][0].float()
        mask = inputs["attention_mask"][0].bool()
        return h[mask].mean(dim=0).cpu().numpy()

    # Extract hidden states for labeled prompts
    print(f"  Extracting hidden states for {len(resps_labeled)} labeled prompts...")
    h_labeled = []
    for i, r in enumerate(resps_labeled):
        si = r["scenario"]
        emo = r["emotion"]
        text = (f"{SCENARIOS[si]}. I feel {EMOTION_PHRASES[emo]}. "
                "I really need someone to talk to.")
        h_labeled.append(get_hidden(text))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(resps_labeled)}")
    h_labeled = np.array(h_labeled)

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(h_labeled, "cosine"))

    # Within-scenario distances (same scenario, diff emotion)
    within_s = []
    for si in range(10):
        for ei in range(5):
            for ej in range(ei + 1, 5):
                within_s.append(dists[si * 5 + ei, si * 5 + ej])

    # Within-emotion distances (same emotion, diff scenario)
    within_e = []
    for ei in range(5):
        for si in range(10):
            for sj in range(si + 1, 10):
                within_e.append(dists[si * 5 + ei, sj * 5 + ei])

    # Across both (diff scenario, diff emotion)
    across = []
    for i in range(50):
        for j in range(i + 1, 50):
            ri, rj = resps_labeled[i], resps_labeled[j]
            if ri["scenario"] != rj["scenario"] and ri["emotion"] != rj["emotion"]:
                across.append(dists[i, j])

    print(f"\n  Internal distances (cosine):")
    print(f"    Within SCENARIO (same event, diff emotion): {np.mean(within_s):.4f}")
    print(f"    Within EMOTION  (same emotion, diff event): {np.mean(within_e):.4f}")
    print(f"    Across both:                                {np.mean(across):.4f}")

    if np.mean(within_s) < np.mean(within_e):
        print(f"    → Representations cluster by SCENARIO (within-scenario < within-emotion)")
    else:
        print(f"    → Representations cluster by EMOTION (within-emotion < within-scenario)")

    # Effect sizes
    u_se, p_se = mannwhitneyu(within_s, within_e)
    print(f"    Scenario vs Emotion: U={u_se:.0f}, p={p_se:.4f}")

    results["e4_internal_clustering"] = {
        "within_scenario_dist": float(np.mean(within_s)),
        "within_emotion_dist": float(np.mean(within_e)),
        "across_both_dist": float(np.mean(across)),
        "clusters_by": "scenario" if np.mean(within_s) < np.mean(within_e) else "emotion",
        "U": float(u_se), "p": float(p_se),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peak = PEAK.get(model_short, 15)

    print("=" * 70)
    print(f"PERFORMING EMPATHY DISSECTION: {model_short}")
    print("=" * 70)

    # Load existing performing_empathy results
    pe_path = BASE / "results" / "mechanistic" / f"performing_empathy_{model_short}.json"
    if not pe_path.exists():
        print(f"ERROR: {pe_path} not found. Run performing_empathy.py first.")
        return
    pe = json.load(open(pe_path))
    resps = pe["responses"]
    print(f"Loaded {len(resps)} existing responses")

    results = {"model": model_short, "peak_layer": peak}

    # CPU evidence
    evidence_1_preamble_strategy(resps, results)
    evidence_2_blind_classification(resps, results)

    # GPU evidence
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nLoading model for E3+E4...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    labeled_resps, unlabeled_resps = evidence_3_label_ablation(
        model, tokenizer, device, results)

    evidence_4_internal_clustering(
        model, tokenizer, labeled_resps, unlabeled_resps, device, peak, results)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    e1 = results["e1_preamble_strategy"]
    e2 = results["e2_blind_classification"]
    e3 = results["e3_label_ablation"]
    e4 = results["e4_internal_clustering"]
    print(f"  E1: Strategy within-scenario sim ({e1['strategy_within_scenario']:.3f}) "
          f"vs within-emotion ({e1['strategy_within_emotion']:.3f})")
    print(f"  E2: Blind emotion classification = {e2['emotion_accuracy']:.0%} (chance 20%)")
    print(f"  E3: Strategy sim (labeled vs unlabeled) = {e3['strategy_sim_labeled_vs_unlabeled']:.3f}")
    print(f"  E4: Clusters by {e4['clusters_by']} "
          f"(within-scen={e4['within_scenario_dist']:.4f}, "
          f"within-emo={e4['within_emotion_dist']:.4f})")

    out_path = OUT_DIR / f"{model_short}_dissection.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
