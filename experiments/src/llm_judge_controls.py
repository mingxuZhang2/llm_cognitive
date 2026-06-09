#!/usr/bin/env python3
"""
LLM judge on control steering directions (random/sentiment/PC1) vs brain axis.
Compares whether each direction produces perceptible emotional↔analytical shifts.

If only brain-derived axis gives significant ρ, that confirms specificity:
the alignment is not an artifact of any arbitrary perturbation.
"""
from __future__ import annotations
import json, os, random, time, sys, re
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
from scipy.stats import spearmanr, ttest_1samp, wilcoxon, mannwhitneyu

BASE = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE / "results" / "human_rating"

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    sys.exit("ERROR: set DEEPSEEK_API_KEY environment variable")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
SEED = 20260604


def call_deepseek(prompt, max_tokens=200, temperature=0.0):
    for attempt in range(3):
        try:
            resp = requests.post(API_URL,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens,
                      "temperature": temperature},
                timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            print(f"    API status {resp.status_code}, retry {attempt+1}")
            time.sleep(2)
        except Exception as e:
            print(f"    API error: {e}, retry {attempt+1}")
            time.sleep(2)
    return None


def build_ranking_prompt(original_prompt, responses_shuffled, labels):
    resp_text = ""
    for label, resp in zip(labels, responses_shuffled):
        truncated = resp[:800] if len(resp) > 800 else resp
        resp_text += f"\n**Response {label}:**\n{truncated}\n"

    return f"""You are evaluating 5 different responses to the same prompt. Your task is to rank them from MOST EMOTIONAL/EMPATHETIC to MOST ANALYTICAL/DETACHED.

**The prompt given to the AI:**
{original_prompt[:500]}

**The 5 responses (labeled A through E):**
{resp_text}

**Instructions:**
- Rank all 5 responses from most emotional/empathetic (rank 1) to most analytical/detached (rank 5).
- Consider: emotional language, personal warmth, empathy, vs logical structure, analytical frameworks, detached reasoning.
- Output ONLY a JSON object like: {{"ranking": ["C", "A", "E", "B", "D"]}}
  where the first letter is the most emotional and the last is the most analytical.
- Do not explain. Just output the JSON."""


def parse_ranking(response_text):
    try:
        match = re.search(r'\{[^}]*"ranking"\s*:\s*\[([^\]]+)\]', response_text)
        if match:
            letters = re.findall(r'"([A-E])"', match.group(1))
            if len(letters) == 5 and set(letters) == {"A", "B", "C", "D", "E"}:
                return letters
        letters = re.findall(r'\b([A-E])\b', response_text)
        seen = []
        for l in letters:
            if l not in seen:
                seen.append(l)
        if len(seen) == 5:
            return seen
    except:
        pass
    return None


def judge_one_direction(name, responses, rng):
    """Run DeepSeek judge on one steering direction. Returns list of per-prompt ρ values."""
    by_prompt = defaultdict(list)
    for r in responses:
        by_prompt[r["prompt_id"]].append(r)

    print(f"\n{'='*60}")
    print(f"  Direction: {name} ({len(responses)} responses, {len(by_prompt)} prompts)")
    print(f"{'='*60}")

    all_rhos = []

    for pi, (pid, resps) in enumerate(sorted(by_prompt.items())):
        resps_sorted = sorted(resps, key=lambda x: x["alpha"])
        if len(resps_sorted) != 5:
            print(f"  [{pi+1:2d}] {pid}: expected 5 alphas, got {len(resps_sorted)}, SKIP")
            continue

        indices = list(range(5))
        rng.shuffle(indices)
        shuffled = [resps_sorted[i] for i in indices]
        labels = ["A", "B", "C", "D", "E"]

        label_to_alpha = {}
        for li, idx in enumerate(indices):
            label_to_alpha[labels[li]] = resps_sorted[idx]["alpha"]

        prompt_text = build_ranking_prompt(
            resps[0]["prompt"], [s["response"] for s in shuffled], labels)

        response = call_deepseek(prompt_text)
        ranking = parse_ranking(response) if response else None

        if ranking:
            rank_by_alpha = {}
            for rank_pos, label in enumerate(ranking):
                alpha = label_to_alpha[label]
                rank_by_alpha[alpha] = rank_pos + 1

            alphas_ordered = sorted(rank_by_alpha.keys())
            ranks_ordered = [rank_by_alpha[a] for a in alphas_ordered]
            rho, _ = spearmanr(alphas_ordered, ranks_ordered)

            status = "+" if rho > 0.3 else ("-" if rho < -0.3 else "~")
            print(f"  [{pi+1:2d}/30] {pid:12s} ρ={rho:+.2f} {status}")
            all_rhos.append(rho)
        else:
            print(f"  [{pi+1:2d}/30] {pid:12s} PARSE FAILED")

        time.sleep(0.5)

    return all_rhos


def main():
    rng = random.Random(SEED)

    directions = {}

    # Brain axis (already judged, load from saved results)
    brain_path = RESULTS_DIR / "deepseek_judge_ranking.json"
    if brain_path.exists():
        brain_data = json.load(open(brain_path))
        brain_rhos = [r["rho"] for r in brain_data["per_prompt"] if r["rho"] is not None]
        directions["brain"] = brain_rhos
        print(f"Brain axis (loaded): n={len(brain_rhos)}, mean ρ={np.mean(brain_rhos):+.3f}")
    else:
        print("WARNING: brain axis results not found, will skip comparison")

    # Control directions
    control_files = {
        "random": "random_rating_responses.json",
        "sentiment": "sentiment_rating_responses.json",
        "pc1": "pc1_rating_responses.json",
    }

    for ctrl_name, fname in control_files.items():
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping {ctrl_name}")
            continue

        responses = json.load(open(fpath))
        rhos = judge_one_direction(ctrl_name, responses, rng)
        directions[ctrl_name] = rhos

    # Summary
    print(f"\n{'='*60}")
    print(f"COMPARISON: Brain axis vs control directions")
    print(f"{'='*60}")
    print(f"{'Direction':15s} {'n':>4s} {'mean ρ':>8s} {'std':>6s} {'t-test p':>10s} {'sig?':>5s}")
    print("-" * 52)

    for name, rhos in directions.items():
        if len(rhos) < 3:
            print(f"{name:15s} {len(rhos):4d}  (too few)")
            continue
        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos)
        t_stat, t_p = ttest_1samp(rhos, 0)
        sig = "YES" if t_p < 0.05 else "no"
        print(f"{name:15s} {len(rhos):4d} {mean_rho:+8.3f} {std_rho:6.3f} {t_p:10.4f} {sig:>5s}")

    # Brain vs each control: Mann-Whitney U
    if "brain" in directions:
        brain_rhos = directions["brain"]
        print(f"\nBrain vs control (Mann-Whitney U):")
        for ctrl_name in ["random", "sentiment", "pc1"]:
            if ctrl_name not in directions or len(directions[ctrl_name]) < 3:
                continue
            ctrl_rhos = directions[ctrl_name]
            u_stat, u_p = mannwhitneyu(brain_rhos, ctrl_rhos, alternative="greater")
            diff = np.mean(brain_rhos) - np.mean(ctrl_rhos)
            print(f"  brain vs {ctrl_name:10s}: Δρ={diff:+.3f}, U={u_stat:.0f}, p={u_p:.4f}"
                  f" {'*** BRAIN WINS' if u_p < 0.05 else ''}")

    # Save
    out = {
        "judge_model": MODEL,
        "target_model": "Qwen2.5-7B-Instruct",
        "directions": {},
    }
    for name, rhos in directions.items():
        if len(rhos) >= 3:
            t_stat, t_p = ttest_1samp(rhos, 0)
            out["directions"][name] = {
                "n": len(rhos),
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "median_rho": float(np.median(rhos)),
                "ttest_p": float(t_p),
                "per_prompt_rhos": [float(r) for r in rhos],
            }

    if "brain" in directions:
        for ctrl_name in ["random", "sentiment", "pc1"]:
            if ctrl_name in directions and len(directions[ctrl_name]) >= 3:
                u_stat, u_p = mannwhitneyu(
                    directions["brain"], directions[ctrl_name], alternative="greater")
                out["directions"][ctrl_name]["vs_brain_U"] = float(u_stat)
                out["directions"][ctrl_name]["vs_brain_p"] = float(u_p)

    out_path = RESULTS_DIR / "deepseek_judge_controls.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
