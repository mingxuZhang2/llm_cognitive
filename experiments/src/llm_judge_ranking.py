#!/usr/bin/env python3
"""
LLM judge ranking: use DeepSeek to rank steered responses from most emotional
to most analytical, blind to steering condition. Validates that the brain-derived
axis produces human/LLM-perceptible behavioral differences.

Uses DeepSeek API (OpenAI-compatible) to judge Qwen outputs — different company,
different architecture, not "LLM judging itself."
"""
from __future__ import annotations
import json, os, random, time, sys
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
from scipy.stats import spearmanr, wilcoxon

BASE = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE / "results" / "human_rating"

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-5e64e17d84d649bc9f2d86bc1839728a")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
SEED = 20260603

ALPHAS = [-10, -5, 0, 5, 10]


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
            print(f"  API status {resp.status_code}, retry {attempt+1}")
            time.sleep(2)
        except Exception as e:
            print(f"  API error: {e}, retry {attempt+1}")
            time.sleep(2)
    return None


def build_ranking_prompt(task_type, original_prompt, responses_shuffled, labels):
    """Build the prompt asking DeepSeek to rank 5 responses."""
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
    """Extract ranking from DeepSeek response."""
    try:
        # Try direct JSON parse
        import re
        match = re.search(r'\{[^}]*"ranking"\s*:\s*\[([^\]]+)\]', response_text)
        if match:
            letters = re.findall(r'"([A-E])"', match.group(1))
            if len(letters) == 5 and set(letters) == {"A", "B", "C", "D", "E"}:
                return letters
        # Fallback: look for any sequence of 5 unique letters
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


def main():
    # Load brain-axis responses
    resp_path = RESULTS_DIR / "Qwen2.5-7B-Instruct_rating_responses.json"
    if not resp_path.exists():
        print(f"ERROR: {resp_path} not found")
        sys.exit(1)

    results = json.load(open(resp_path))

    # Group by prompt
    by_prompt = defaultdict(list)
    for r in results:
        by_prompt[r["prompt_id"]].append(r)

    print(f"Loaded {len(results)} responses for {len(by_prompt)} prompts")
    print(f"Ranking with DeepSeek ({MODEL})...\n")

    random.seed(SEED + 77)

    all_rankings = []

    for pi, (pid, resps) in enumerate(sorted(by_prompt.items())):
        resps_sorted = sorted(resps, key=lambda x: x["alpha"])
        assert len(resps_sorted) == 5, f"{pid}: expected 5, got {len(resps_sorted)}"

        # Shuffle for blind ranking
        indices = list(range(5))
        random.shuffle(indices)
        shuffled = [resps_sorted[i] for i in indices]
        labels = ["A", "B", "C", "D", "E"]

        # Map: label -> alpha
        label_to_alpha = {}
        for li, idx in enumerate(indices):
            label_to_alpha[labels[li]] = resps_sorted[idx]["alpha"]

        prompt_text = build_ranking_prompt(
            resps[0]["task"], resps[0]["prompt"],
            [s["response"] for s in shuffled], labels)

        response = call_deepseek(prompt_text)
        ranking = parse_ranking(response) if response else None

        if ranking:
            # Convert ranking to alpha order
            # ranking[0] = most emotional, ranking[4] = most analytical
            alpha_ranking = [label_to_alpha[l] for l in ranking]
            # Assign rank positions: rank 1 = most emotional, rank 5 = most analytical
            rank_by_alpha = {}
            for rank_pos, label in enumerate(ranking):
                alpha = label_to_alpha[label]
                rank_by_alpha[alpha] = rank_pos + 1  # 1-indexed

            # Spearman: do higher alphas get higher ranks (more analytical)?
            # If steering works: alpha=-10 should be rank 5 (most analytical)
            # alpha=+10 should be rank 1 (most emotional)
            # So we expect NEGATIVE correlation (higher alpha = more emotional = lower rank number)
            alphas_ordered = sorted(rank_by_alpha.keys())
            ranks_ordered = [rank_by_alpha[a] for a in alphas_ordered]

            rho, _ = spearmanr(alphas_ordered, ranks_ordered)

            status = "✓" if rho < -0.3 else ("~" if abs(rho) < 0.3 else "✗")
            print(f"  [{pi+1:2d}/30] {pid:12s} ({resps[0]['task']:15s}) "
                  f"ρ(α,rank)={rho:+.2f} {status}  "
                  f"ranking: {' '.join(ranking)}")
        else:
            rho = None
            print(f"  [{pi+1:2d}/30] {pid:12s} PARSE FAILED: {response[:80] if response else 'NO RESPONSE'}")

        all_rankings.append({
            "prompt_id": pid,
            "task": resps[0]["task"],
            "label_to_alpha": label_to_alpha,
            "ranking": ranking,
            "rho": rho,
            "raw_response": response,
        })

        time.sleep(0.5)  # rate limit

    # Summary statistics
    valid = [r for r in all_rankings if r["rho"] is not None]
    rhos = [r["rho"] for r in valid]

    print(f"\n{'='*60}")
    print(f"SUMMARY: DeepSeek LLM Judge Ranking")
    print(f"{'='*60}")
    print(f"Valid rankings: {len(valid)}/{len(all_rankings)}")
    print(f"Mean ρ(alpha, rank): {np.mean(rhos):+.3f} ± {np.std(rhos):.3f}")
    print(f"Median ρ: {np.median(rhos):+.3f}")

    # One-sample t-test / Wilcoxon: is mean ρ significantly != 0?
    if len(rhos) >= 5:
        from scipy.stats import ttest_1samp
        t_stat, t_p = ttest_1samp(rhos, 0)
        try:
            w_stat, w_p = wilcoxon(rhos)
        except:
            w_stat, w_p = None, None
        print(f"t-test vs 0: t={t_stat:.3f}, p={t_p:.4f}")
        if w_p is not None:
            print(f"Wilcoxon vs 0: W={w_stat:.0f}, p={w_p:.4f}")

    # Per-task breakdown
    for task in ["moral", "empathy", "tom_faux_pas"]:
        task_rhos = [r["rho"] for r in valid if r["task"] == task]
        if task_rhos:
            print(f"  {task:15s}: mean ρ={np.mean(task_rhos):+.3f} (n={len(task_rhos)})")

    # Prediction check
    # Direction = mentalistic - affective, so positive alpha = toward analytical
    # Ranking: rank 1 = most emotional, rank 5 = most analytical
    # Expected: POSITIVE correlation (higher alpha → higher rank → more analytical)
    print(f"\nPREDICTION: brain direction = ment-aff, so positive alpha = more analytical")
    print(f"  → ρ(alpha, analytical_rank) should be POSITIVE")
    print(f"  → Mean ρ = {np.mean(rhos):+.3f}")
    if np.mean(rhos) > 0.1 and t_p < 0.05:
        print(f"  → CONFIRMED: brain-derived axis produces LLM-perceptible behavioral shift (p={t_p:.4f})")
    elif np.mean(rhos) > 0:
        print(f"  → TREND in predicted direction but {'not significant' if t_p > 0.05 else 'marginal'}")
    else:
        print(f"  → NOT CONFIRMED")

    # Save
    out_path = RESULTS_DIR / "deepseek_judge_ranking.json"
    out = {
        "judge_model": MODEL,
        "target_model": "Qwen2.5-7B-Instruct",
        "n_prompts": len(all_rankings),
        "n_valid": len(valid),
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos)),
        "median_rho": float(np.median(rhos)),
        "ttest_p": float(t_p) if len(rhos) >= 5 else None,
        "wilcoxon_p": float(w_p) if w_p is not None else None,
        "per_prompt": all_rankings,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
