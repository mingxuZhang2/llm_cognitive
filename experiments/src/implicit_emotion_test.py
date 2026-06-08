#!/usr/bin/env python3
"""
Implicit vs explicit emotion test: is LLM emotion differentiation lexical or
representational?

If lexical: model distinguishes emotions when emotion WORDS are present,
but NOT when emotions are only IMPLIED through descriptions.

If representational: model distinguishes even without emotion words.

Design:
  - 10 scenarios x 5 emotions x 2 conditions (explicit vs implicit)
  - Explicit: "I feel angry about X"
  - Implicit: describes the EXPERIENCE without naming the emotion
    ("my fists clenched, I wanted to confront them")
  - Measure: JSD between emotion pairs in explicit vs implicit conditions
  - Prediction: explicit JSD > implicit JSD for negative emotion pairs
    (model can distinguish when told which emotion, but not from description alone)
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
from scipy.spatial.distance import jensenshannon

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
OUT_DIR.mkdir(exist_ok=True)

EMOTIONS = ["anger", "fear", "disgust", "sadness", "happiness"]

# Each scenario has an explicit version (names the emotion) and an implicit
# version (describes the bodily/behavioral experience without naming it)
SCENARIOS = [
    {
        "context": "My best friend lied to me about something important.",
        "explicit": {
            "anger":     "I feel so angry about this. What should I do?",
            "fear":      "I feel so afraid about this. What should I do?",
            "disgust":   "I feel so disgusted by this. What should I do?",
            "sadness":   "I feel so sad about this. What should I do?",
            "happiness": "I actually feel relieved about this. What should I do?",
        },
        "implicit": {
            "anger":     "My jaw is clenched and I keep replaying what happened, wanting to confront them. What should I do?",
            "fear":      "I can't stop worrying about what this means for our friendship. My mind is racing with worst-case scenarios. What should I do?",
            "disgust":   "I feel sick to my stomach just thinking about it. I don't even want to be around them anymore. What should I do?",
            "sadness":   "I keep tearing up randomly. Everything reminds me of how things used to be between us. What should I do?",
            "happiness": "I feel like a weight has been lifted. I finally see them clearly now. What should I do?",
        },
    },
    {
        "context": "I found out my coworker has been taking credit for my work.",
        "explicit": {
            "anger":     "I'm furious about this situation. What should I do?",
            "fear":      "I'm terrified this will hurt my career. What should I do?",
            "disgust":   "I'm repulsed by their behavior. What should I do?",
            "sadness":   "I feel heartbroken and betrayed. What should I do?",
            "happiness": "I'm oddly calm about it, like I expected this. What should I do?",
        },
        "implicit": {
            "anger":     "I can barely sit still. I want to march into my manager's office right now and expose them. What should I do?",
            "fear":      "I keep checking if my other projects are safe. I couldn't sleep last night thinking about what else they might take. What should I do?",
            "disgust":   "Every time I see them smile at the meeting, my stomach turns. I can't stand being in the same room. What should I do?",
            "sadness":   "I spent years building that work. Now it feels like it was all for nothing. I just stare at my screen. What should I do?",
            "happiness": "Honestly, this just confirmed what I suspected. I feel clear-headed about my next move. What should I do?",
        },
    },
    {
        "context": "Someone was incredibly rude to an elderly person on the bus.",
        "explicit": {
            "anger":     "I'm so angry about what I saw. What should I do?",
            "fear":      "I'm scared that society is becoming so cruel. What should I do?",
            "disgust":   "I'm disgusted by that person's behavior. What should I do?",
            "sadness":   "I feel so sad for that elderly person. What should I do?",
            "happiness": "I'm glad I was there to help. What should I do?",
        },
        "implicit": {
            "anger":     "I could feel my face getting hot. I wanted to stand up and tell that person off right there. What should I do?",
            "fear":      "It made me think about my own parents. What if nobody stands up for them when they need it? What should I do?",
            "disgust":   "I had to look away. The way they spoke to that person made my skin crawl. What should I do?",
            "sadness":   "The look on the elderly person's face broke my heart. They just went quiet and looked at the floor. What should I do?",
            "happiness": "I stepped in and helped, and the elderly person smiled at me. That moment stayed with me all day. What should I do?",
        },
    },
    {
        "context": "My partner forgot our anniversary.",
        "explicit": {
            "anger":     "I'm really angry about this. What should I do?",
            "fear":      "I'm afraid this means they don't care anymore. What should I do?",
            "disgust":   "I'm disgusted that they could be so thoughtless. What should I do?",
            "sadness":   "I feel so sad and unimportant. What should I do?",
            "happiness": "I actually don't mind. We had a nice quiet evening instead. What should I do?",
        },
        "implicit": {
            "anger":     "I kept bringing it up all evening. I couldn't let it go. Every little thing they did annoyed me more. What should I do?",
            "fear":      "I lay awake wondering if this is the beginning of the end. Are we slowly falling apart? What should I do?",
            "disgust":   "I looked at them and felt nothing. Like they're a stranger who doesn't know me at all. What should I do?",
            "sadness":   "I sat alone in the kitchen after they fell asleep, looking at old photos of us on my phone. What should I do?",
            "happiness": "We just had takeout and watched a movie together. It was actually really nice. What should I do?",
        },
    },
    {
        "context": "I was passed over for a promotion at work.",
        "explicit": {
            "anger":     "I'm absolutely livid. What should I do?",
            "fear":      "I'm scared my career is going nowhere. What should I do?",
            "disgust":   "I'm disgusted by the whole process. What should I do?",
            "sadness":   "I feel defeated and sad. What should I do?",
            "happiness": "Honestly I'm relieved. That role would have been too much stress. What should I do?",
        },
        "implicit": {
            "anger":     "I almost quit on the spot. I've been grinding for years while others coast by. My blood is boiling. What should I do?",
            "fear":      "I keep wondering if I should start looking for other jobs. What if I'm stuck here forever? What should I do?",
            "disgust":   "The person they chose instead... I can't respect the decision. The whole system feels corrupt. What should I do?",
            "sadness":   "I went to the bathroom and cried. All that effort, all those late nights, for nothing. What should I do?",
            "happiness": "Now I can focus on what I actually enjoy without the management headaches. What should I do?",
        },
    },
]


def get_next_token_dist(model, tokenizer, text, device, top_k=1000):
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1).cpu()
    topk_probs, topk_ids = torch.topk(probs, top_k)
    dist = torch.zeros(probs.shape[0])
    dist[topk_ids] = topk_probs
    dist = dist / dist.sum()
    return dist.numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Implicit vs Explicit Emotion Test ({model_short}) ===")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_scenarios = len(SCENARIOS)
    pairs = list(itertools.combinations(EMOTIONS, 2))

    # Compute JSD for each pair x scenario x condition (explicit/implicit)
    explicit_jsds = {p: [] for p in pairs}
    implicit_jsds = {p: [] for p in pairs}

    for si, sc in enumerate(SCENARIOS):
        print(f"\n  Scenario {si+1}/{n_scenarios}: {sc['context'][:50]}...")

        # Get distributions for explicit and implicit
        exp_dists = {}
        imp_dists = {}
        for emo in EMOTIONS:
            exp_dists[emo] = get_next_token_dist(model, tokenizer, sc["explicit"][emo], device)
            imp_dists[emo] = get_next_token_dist(model, tokenizer, sc["implicit"][emo], device)

        # Compute JSD for each pair
        for e1, e2 in pairs:
            ej = float(jensenshannon(exp_dists[e1], exp_dists[e2]))
            ij = float(jensenshannon(imp_dists[e1], imp_dists[e2]))
            if not np.isnan(ej):
                explicit_jsds[(e1, e2)].append(ej)
            if not np.isnan(ij):
                implicit_jsds[(e1, e2)].append(ij)

    # ─── Results ───
    print(f"\n{'='*80}")
    print("RESULTS: Explicit vs Implicit emotion differentiation")
    print(f"{'='*80}")

    # Separate negative-only pairs from positive-negative pairs
    neg_emotions = ["anger", "fear", "disgust", "sadness"]
    neg_pairs = [(e1, e2) for e1, e2 in pairs if e1 in neg_emotions and e2 in neg_emotions]
    pos_neg_pairs = [(e1, e2) for e1, e2 in pairs if e1 not in neg_emotions or e2 not in neg_emotions]

    print(f"\n--- Per-pair JSD (explicit vs implicit) ---")
    print(f"{'Pair':>25s} | {'Explicit JSD':>12s} | {'Implicit JSD':>12s} | {'Drop':>8s} | {'Type':>8s}")
    print("-" * 75)

    all_exp_neg = []
    all_imp_neg = []
    all_exp_posneg = []
    all_imp_posneg = []

    for e1, e2 in pairs:
        exp_mean = np.mean(explicit_jsds[(e1, e2)])
        imp_mean = np.mean(implicit_jsds[(e1, e2)])
        drop = (exp_mean - imp_mean) / exp_mean * 100 if exp_mean > 0 else 0
        ptype = "neg-neg" if (e1, e2) in neg_pairs else "pos-neg"
        print(f"{e1+'-'+e2:>25s} | {exp_mean:12.5f} | {imp_mean:12.5f} | {drop:+7.1f}% | {ptype:>8s}")

        if (e1, e2) in neg_pairs:
            all_exp_neg.extend(explicit_jsds[(e1, e2)])
            all_imp_neg.extend(implicit_jsds[(e1, e2)])
        else:
            all_exp_posneg.extend(explicit_jsds[(e1, e2)])
            all_imp_posneg.extend(implicit_jsds[(e1, e2)])

    # Key test: does implicit reduce differentiation among negative emotions?
    print(f"\n--- KEY TEST: Negative emotion pairs ---")
    exp_neg_mean = np.mean(all_exp_neg)
    imp_neg_mean = np.mean(all_imp_neg)
    print(f"  Explicit: mean JSD = {exp_neg_mean:.5f} (n={len(all_exp_neg)})")
    print(f"  Implicit: mean JSD = {imp_neg_mean:.5f} (n={len(all_imp_neg)})")
    print(f"  Change: {(imp_neg_mean - exp_neg_mean) / exp_neg_mean * 100:+.1f}%")

    # Paired test across scenarios
    # For each scenario, compute mean neg-pair JSD explicit vs implicit
    scenario_exp = []
    scenario_imp = []
    for si in range(n_scenarios):
        exp_vals = [explicit_jsds[p][si] for p in neg_pairs if si < len(explicit_jsds[p])]
        imp_vals = [implicit_jsds[p][si] for p in neg_pairs if si < len(implicit_jsds[p])]
        if exp_vals and imp_vals:
            scenario_exp.append(np.mean(exp_vals))
            scenario_imp.append(np.mean(imp_vals))

    scenario_exp = np.array(scenario_exp)
    scenario_imp = np.array(scenario_imp)

    if len(scenario_exp) >= 5:
        w, pw = wilcoxon(scenario_exp - scenario_imp)
        print(f"  Wilcoxon (explicit > implicit): W={w:.0f}, p={pw:.4f}")
        n_exp_higher = np.sum(scenario_exp > scenario_imp)
        print(f"  Explicit > Implicit in {n_exp_higher}/{len(scenario_exp)} scenarios")

    # Same for pos-neg pairs
    print(f"\n--- CONTROL: Positive-negative pairs ---")
    exp_pn_mean = np.mean(all_exp_posneg)
    imp_pn_mean = np.mean(all_imp_posneg)
    print(f"  Explicit: mean JSD = {exp_pn_mean:.5f}")
    print(f"  Implicit: mean JSD = {imp_pn_mean:.5f}")
    print(f"  Change: {(imp_pn_mean - exp_pn_mean) / exp_pn_mean * 100:+.1f}%")

    # Interaction: does the explicit→implicit drop differ for neg-neg vs pos-neg?
    print(f"\n--- INTERACTION ---")
    neg_drop = imp_neg_mean - exp_neg_mean
    pn_drop = imp_pn_mean - exp_pn_mean
    print(f"  Neg-neg JSD change: {neg_drop:+.5f}")
    print(f"  Pos-neg JSD change: {pn_drop:+.5f}")
    print(f"  If neg-neg drops MORE, model's neg differentiation is lexical")

    # Save
    results = {
        "model": model_short,
        "n_scenarios": n_scenarios,
        "neg_pairs_explicit_jsd": float(exp_neg_mean),
        "neg_pairs_implicit_jsd": float(imp_neg_mean),
        "posneg_pairs_explicit_jsd": float(exp_pn_mean),
        "posneg_pairs_implicit_jsd": float(imp_pn_mean),
        "per_pair": {
            f"{e1}-{e2}": {
                "explicit_mean": float(np.mean(explicit_jsds[(e1, e2)])),
                "implicit_mean": float(np.mean(implicit_jsds[(e1, e2)])),
                "explicit_vals": [float(v) for v in explicit_jsds[(e1, e2)]],
                "implicit_vals": [float(v) for v in implicit_jsds[(e1, e2)]],
            }
            for e1, e2 in pairs
        },
    }

    if len(scenario_exp) >= 5:
        results["wilcoxon_neg_pairs"] = {"W": float(w), "p": float(pw)}

    out_path = OUT_DIR / f"implicit_emotion_test_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
