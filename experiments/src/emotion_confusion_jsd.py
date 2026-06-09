#!/usr/bin/env python3
"""
Emotion confusion test v2: does LLM emotion geometry predict response confusion?

Uses next-token distribution JSD (Jensen-Shannon divergence) instead of
embedding cosine — much more sensitive to how the model's predictions change
when the emotion label changes.

For each scenario x emotion pair: compute JSD between next-token distributions.
Low JSD = model predicts same things = model conflates these emotions.
High JSD = model predicts different things = model distinguishes them.

Correlate JSD with LLM emotion RDM distances and brain distances.

GPU needed.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
OUT_DIR.mkdir(exist_ok=True)

AFF = ["anger", "fear", "disgust", "sadness", "happiness"]

SCENARIOS = [
    "My best friend lied to me about something important.",
    "I found out my coworker has been taking credit for my work.",
    "Someone cut in line right in front of me after I'd been waiting for an hour.",
    "A company has been polluting the river near my hometown.",
    "My partner forgot our anniversary even though I reminded them twice.",
    "I watched a documentary about animal cruelty in factory farms.",
    "A stranger attacked me personally for sharing my opinion.",
    "I learned that my childhood home is being demolished.",
    "My doctor told me my test results were concerning.",
    "Someone was incredibly rude to an elderly person on the bus.",
    "I discovered my neighbor has been stealing my packages.",
    "My child was bullied at school and nobody intervened.",
    "I was passed over for a promotion that went to someone less qualified.",
    "A close family member said something deeply hurtful during an argument.",
    "I found out a charity I donated to was actually a scam.",
    "My landlord is refusing to fix a serious safety hazard.",
    "Someone spread false rumors about me at work.",
    "I witnessed a hit and run accident and the driver fled.",
    "My therapist broke confidentiality and told someone about my sessions.",
    "A trusted mentor plagiarized my research and published it.",
    "I received news that a beloved pet is terminally ill.",
    "My flight was canceled and the airline refused to help.",
    "I found evidence that a local official is corrupt.",
    "A friend borrowed a large sum of money and now avoids me.",
    "I accidentally overheard my colleagues mocking me behind my back.",
    "Someone vandalized my car in the parking lot.",
    "I learned my insurance claim was denied on a technicality.",
    "A restaurant served me food I'm allergic to after I warned them.",
    "My sibling revealed a family secret that changes everything.",
    "I was falsely accused of something I didn't do.",
]

EMOTION_TEMPLATES = {
    "anger": "This makes me feel really angry.",
    "fear": "This makes me feel really afraid.",
    "disgust": "This makes me feel really disgusted.",
    "sadness": "This makes me feel really sad.",
    "happiness": "Surprisingly, this makes me feel oddly relieved.",
}


def get_next_token_dist(model, tokenizer, text, device, top_k=1000):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
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

    print(f"=== Emotion Confusion JSD Test ({model_short}) ===")
    print(f"Device: {device}")
    print(f"{len(SCENARIOS)} scenarios x {len(AFF)} emotions")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Step 1: Get next-token distributions
    print("\n--- Computing next-token distributions ---")
    dists = {}
    for si, scenario in enumerate(SCENARIOS):
        for emo in AFF:
            prompt = f"{scenario} {EMOTION_TEMPLATES[emo]} What should I do?"
            dist = get_next_token_dist(model, tokenizer, prompt, device)
            dists[(si, emo)] = dist
        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(SCENARIOS)}]")

    # Step 2: JSD per emotion pair
    print("\n--- Computing JSD per emotion pair ---")
    pair_jsds = {}
    for e1, e2 in itertools.combinations(AFF, 2):
        jsds = []
        for si in range(len(SCENARIOS)):
            d1 = dists[(si, e1)]
            d2 = dists[(si, e2)]
            j = float(jensenshannon(d1, d2))
            if not np.isnan(j):
                jsds.append(j)
        pair_jsds[(e1, e2)] = {
            "mean_jsd": float(np.mean(jsds)),
            "std": float(np.std(jsds)),
            "n": len(jsds),
        }

    # Step 3: Load RDMs and correlate
    print("\n--- Correlating with geometry ---")
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_conds = list(ns["conditions"])
    brain_rdm = ns["rdm"]

    rdm_path = RSA / f"{model_short}_rdm14_headline.npz"
    if rdm_path.exists():
        z = np.load(rdm_path, allow_pickle=True)
        llm_conds = list(z["conditions"])
        llm_rdm = z["rdm"]
    else:
        llm_rdm = None
        llm_conds = None

    pair_keys = list(itertools.combinations(AFF, 2))
    jsd_vec = np.array([pair_jsds[k]["mean_jsd"] for k in pair_keys])

    brain_dist_vec = np.array([
        brain_rdm[brain_conds.index(e1), brain_conds.index(e2)]
        for e1, e2 in pair_keys
    ])

    # JSD should POSITIVELY correlate with distance
    # (far pairs → different predictions → high JSD)
    rho_brain, p_brain = spearmanr(jsd_vec, brain_dist_vec)
    print(f"\n  JSD vs brain distance: rho={rho_brain:+.3f} (p={p_brain:.4f})")

    if llm_rdm is not None:
        llm_dist_vec = np.array([
            llm_rdm[llm_conds.index(e1), llm_conds.index(e2)]
            for e1, e2 in pair_keys
        ])
        rho_llm, p_llm = spearmanr(jsd_vec, llm_dist_vec)
        print(f"  JSD vs LLM distance:   rho={rho_llm:+.3f} (p={p_llm:.4f})")
    else:
        rho_llm, p_llm = None, None

    # Step 4: Pair-by-pair results
    print(f"\n{'Pair':>25s} | {'JSD':>8s} | {'Brain':>8s} | {'LLM':>8s} | {'Flag':>10s}")
    print("-" * 70)

    median_jsd = np.median(jsd_vec)
    median_brain = np.median(brain_dist_vec)

    for e1, e2 in sorted(pair_keys, key=lambda k: pair_jsds[k]["mean_jsd"]):
        j = pair_jsds[(e1, e2)]["mean_jsd"]
        bd = brain_rdm[brain_conds.index(e1), brain_conds.index(e2)]
        ld = llm_rdm[llm_conds.index(e1), llm_conds.index(e2)] if llm_rdm is not None else 0

        if j < median_jsd and bd > median_brain:
            flag = "CONFUSE"
        elif j > median_jsd and bd < median_brain:
            flag = "OVER-SEP"
        else:
            flag = ""
        print(f"{e1+'-'+e2:>25s} | {j:8.5f} | {bd:8.4f} | {ld:8.4f} | {flag:>10s}")

    # Save
    results = {
        "model": model_short,
        "n_scenarios": len(SCENARIOS),
        "pair_jsd": {f"{e1}-{e2}": pair_jsds[(e1, e2)] for e1, e2 in pair_keys},
        "correlation_brain": {"rho": float(rho_brain), "p": float(p_brain)},
        "correlation_llm": {"rho": float(rho_llm), "p": float(p_llm)} if rho_llm is not None else None,
    }
    out_path = OUT_DIR / f"emotion_confusion_jsd_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
