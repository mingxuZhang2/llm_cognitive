#!/usr/bin/env python3
"""
Generate steered LLM responses for blind human rating experiment.

Three task types × 10 prompts × 5 steering alphas = 150 responses per model.
Alphas are moderate (-10, -5, 0, +5, +10) to avoid model collapse.

After generation, prints sample responses at each alpha for quality inspection.
Outputs a shuffled rating spreadsheet (TSV) for blind raters.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "human_rating"

ALPHAS = [-10, -5, 0, 5, 10]
SEED = 20260603

# ── Steering (reuse logic from cognitive_steering.py) ────────────────────

AFFECTIVE_TEXTS = [
    "I feel so angry right now, I can barely contain my rage.",
    "The sadness overwhelmed me, tears streaming down uncontrollably.",
    "A wave of fear and terror washed over me completely.",
    "I'm disgusted by what I just witnessed, it makes me sick.",
    "Pure joy and happiness filled my heart in that moment.",
    "The grief was unbearable, a deep aching emotional pain.",
    "I was furious, my blood boiling with intense anger.",
    "A feeling of dread and anxiety consumed my entire being.",
]

MENTALISTIC_TEXTS = [
    "She believed that the meeting would start at three o'clock.",
    "He intended to submit the report before the deadline arrived.",
    "They understood that she was being sarcastic in her remarks.",
    "The judge determined that the evidence was insufficient for conviction.",
    "She realized he didn't know about the surprise party planned.",
    "He inferred from her expression that she disagreed with the proposal.",
    "They concluded that the policy needed to be revised substantially.",
    "She recognized that his perspective differed from her own viewpoint.",
]


def compute_boundary_direction(model, tokenizer, device, peak_layer):
    """Compute the affective→mentalistic boundary direction at peak_layer."""
    def get_mean_hidden(texts):
        vecs = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            h = out.hidden_states[peak_layer][0].mean(0).float().cpu().numpy()
            vecs.append(h)
        return np.mean(vecs, axis=0)

    aff_center = get_mean_hidden(AFFECTIVE_TEXTS)
    ment_center = get_mean_hidden(MENTALISTIC_TEXTS)
    direction = ment_center - aff_center
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction


class SteeringHook:
    def __init__(self, direction, alpha):
        self.direction = torch.tensor(direction, dtype=torch.float16)
        self.alpha = alpha
        self.device_set = False

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if not self.device_set:
            self.direction = self.direction.to(h.device)
            self.device_set = True
        h = h + self.alpha * self.direction
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h


def generate_with_steering(model, tokenizer, prompt, direction, alpha, device,
                           peak_layer, max_new_tokens=200):
    n_layers = model.config.num_hidden_layers
    start_layer = int(n_layers * 0.75)

    hooks = []
    if alpha != 0:
        for i in range(start_layer, n_layers):
            layer_module = model.model.layers[i] if hasattr(model, 'model') else model.transformer.h[i]
            hook = SteeringHook(direction, alpha)
            hooks.append(layer_module.register_forward_hook(hook))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    for h in hooks:
        h.remove()

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    return response.strip()


# ── Task prompts ─────────────────────────────────────────────────────────

def load_moral_prompts(n=10):
    path = BASE / "data" / "cognitive_stimuli" / "moral" / "moral_dilemmas.jsonl"
    items = [json.loads(l) for l in open(path)]
    random.seed(SEED)
    selected = random.sample(items, min(n, len(items)))
    prompts = []
    for d in selected:
        prompt = (f"Consider the following moral dilemma:\n\n{d['scenario']}\n\n"
                  f"What would you do and why? Please explain your reasoning.")
        prompts.append({"id": d["id"], "task": "moral", "prompt": prompt,
                        "type": d.get("subcategory", "unknown")})
    return prompts


def load_empathy_prompts(n=10):
    scenarios = [
        {"id": "emp_01", "situation": "I just got fired from my job today. I don't know how I'm going to pay rent next month."},
        {"id": "emp_02", "situation": "My mother was just diagnosed with cancer. The doctors say it's stage 3."},
        {"id": "emp_03", "situation": "My partner of 5 years just told me they want to break up. I didn't see it coming at all."},
        {"id": "emp_04", "situation": "I failed my PhD qualifying exam. I've been working towards this for 3 years and now I might have to leave the program."},
        {"id": "emp_05", "situation": "My best friend passed away in a car accident last week. We grew up together."},
        {"id": "emp_06", "situation": "I found out my coworkers have been talking behind my back and making fun of my accent."},
        {"id": "emp_07", "situation": "My child was bullied at school again today. He came home crying and said he doesn't want to go back."},
        {"id": "emp_08", "situation": "I just found out I'm pregnant but my partner doesn't want children. I don't know what to do."},
        {"id": "emp_09", "situation": "I've been struggling with depression for months. Some days I can barely get out of bed."},
        {"id": "emp_10", "situation": "My elderly father fell and broke his hip. He lives alone and I'm worried he can't take care of himself anymore."},
    ]
    prompts = []
    for s in scenarios[:n]:
        prompt = (f"Someone tells you: \"{s['situation']}\"\n\n"
                  f"How would you respond to this person?")
        prompts.append({"id": s["id"], "task": "empathy", "prompt": prompt,
                        "type": "empathy_support"})
    return prompts


def load_tom_prompts(n=10):
    path = BASE / "data" / "cognitive_stimuli" / "tom" / "faux_pas.jsonl"
    items = [json.loads(l) for l in open(path)]
    random.seed(SEED + 1)
    selected = random.sample(items, min(n, len(items)))
    prompts = []
    for fp in selected:
        prompt = (f"Read this story and answer the questions:\n\n{fp['story']}\n\n"
                  f"1. Did anyone say something they shouldn't have? If so, who and what?\n"
                  f"2. Why did they say it?\n"
                  f"3. How do you think {fp.get('faux_pas_listener', 'the listener')} felt?")
        prompts.append({"id": fp["id"], "task": "tom_faux_pas", "prompt": prompt,
                        "type": "faux_pas"})
    return prompts


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--peak_layer", type=int, required=True)
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--inspect_only", action="store_true",
                        help="Generate only 2 prompts per task for quick inspection")
    args = parser.parse_args()

    alphas = [int(a) for a in args.alphas.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    n_per_task = 2 if args.inspect_only else 10
    moral_prompts = load_moral_prompts(n_per_task)
    empathy_prompts = load_empathy_prompts(n_per_task)
    tom_prompts = load_tom_prompts(n_per_task)
    all_prompts = moral_prompts + empathy_prompts + tom_prompts
    print(f"Tasks: {n_per_task} moral + {n_per_task} empathy + {n_per_task} ToM "
          f"= {len(all_prompts)} prompts × {len(alphas)} alphas "
          f"= {len(all_prompts) * len(alphas)} responses")

    # Load model
    print(f"\nLoading {args.model_short} from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"Model loaded on {device}")

    # Compute boundary direction
    print("Computing boundary direction ...")
    direction = compute_boundary_direction(model, tokenizer, device, args.peak_layer)
    print(f"Direction norm: {np.linalg.norm(direction):.4f}")

    # Generate responses
    results = []
    t0 = time.time()
    for pi, p in enumerate(all_prompts):
        print(f"\n{'='*60}")
        print(f"[{pi+1}/{len(all_prompts)}] {p['task']} / {p['id']}")
        print(f"{'='*60}")
        print(f"PROMPT: {p['prompt'][:100]}...")

        for alpha in alphas:
            response = generate_with_steering(
                model, tokenizer, p["prompt"], direction, alpha, device,
                args.peak_layer, max_new_tokens=args.max_new_tokens)

            results.append({
                "prompt_id": p["id"],
                "task": p["task"],
                "task_type": p["type"],
                "prompt": p["prompt"],
                "alpha": alpha,
                "response": response,
                "model": args.model_short,
            })

            # Print for inspection
            alpha_label = f"α={alpha:+d}"
            trunc = response[:150].replace("\n", " ")
            print(f"  {alpha_label:>8s}: {trunc}{'...' if len(response) > 150 else ''}")

    elapsed = time.time() - t0
    print(f"\n\nGenerated {len(results)} responses in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save full results
    out_json = OUT_DIR / f"{args.model_short}_rating_responses.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")

    # Create shuffled rating TSV for blind raters
    random.seed(SEED + 42)
    shuffled = list(range(len(results)))
    random.shuffle(shuffled)

    tsv_path = OUT_DIR / f"{args.model_short}_rating_sheet.tsv"
    with open(tsv_path, "w") as f:
        f.write("rating_id\ttask\tprompt\tresponse\temotional_intensity_1to7\t"
                "analytical_quality_1to7\tempathy_understanding_1to7\toverall_quality_1to7\n")
        for ri, idx in enumerate(shuffled):
            r = results[idx]
            prompt_clean = r["prompt"].replace("\t", " ").replace("\n", " | ")
            resp_clean = r["response"].replace("\t", " ").replace("\n", " | ")
            f.write(f"R{ri+1:03d}\t{r['task']}\t{prompt_clean}\t{resp_clean}\t\t\t\t\n")
    print(f"Rating sheet: {tsv_path}")

    # Print quality summary: check for degenerate responses
    print(f"\n{'='*60}")
    print("QUALITY CHECK")
    print(f"{'='*60}")
    for alpha in alphas:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        lengths = [len(r["response"]) for r in alpha_results]
        empty = sum(1 for l in lengths if l < 10)
        short = sum(1 for l in lengths if l < 50)
        print(f"  α={alpha:+3d}: n={len(alpha_results)}, "
              f"mean_len={np.mean(lengths):.0f}, "
              f"empty(<10)={empty}, short(<50)={short}")
        if empty > 0:
            print(f"         ⚠ {empty} empty/garbage responses!")


if __name__ == "__main__":
    main()
