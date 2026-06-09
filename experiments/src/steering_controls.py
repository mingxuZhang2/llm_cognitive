#!/usr/bin/env python3
"""
Control-direction steering: prove the brain-derived axis is special.

Generates steered LLM responses using 3 CONTROL directions (not the brain-derived
affect-mentalizing boundary) across the same 30 prompts and 5 alphas used in
generate_rating_responses.py. This lets us compare: does the brain-derived axis
produce qualitatively different steering effects from arbitrary or data-driven axes?

Control conditions:
  1. random    -- random unit vectors in hidden space (5 seeds, take median)
  2. sentiment -- positive vs negative sentiment texts (is the brain axis just valence?)
  3. pc1       -- first principal component of all 712 per-stim activations at peak layer
                  (is the brain axis just the dominant variance direction?)

Also generates:
  - Per-control ranking sheets (same blind format as brain-axis)
  - A combined comparative ranking sheet (brain-axis vs random, extreme alphas, shuffled)

Usage (GPU):
  python steering_controls.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_short Qwen2.5-7B-Instruct \
      --peak_layer 27

  # If per-stim NPZ is too large for login node, precompute PC1:
  python steering_controls.py --precompute_pc1_only \
      --model_short Qwen2.5-7B-Instruct --peak_layer 27
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "human_rating"
RSA_DIR = BASE / "results" / "cognitive_rsa"

ALPHAS = [-10, -5, 0, 5, 10]
SEED = 20260603          # same as generate_rating_responses.py
N_RANDOM_SEEDS = 5       # 5 random directions, take median effect

# ── Prompt loading (same as generate_rating_responses.py) ──────────────────

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


# ── Steering infrastructure (same as generate_rating_responses.py) ─────────

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
            layer_module = (model.model.layers[i] if hasattr(model, 'model')
                           else model.transformer.h[i])
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


# ── Brain-axis direction (same texts as generate_rating_responses.py) ──────

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
    """Compute the affective->mentalistic boundary direction at peak_layer."""
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


# ── Control direction 1: Random ────────────────────────────────────────────

def compute_random_directions(hidden_dim, n_seeds=N_RANDOM_SEEDS):
    """Return n_seeds random unit vectors in hidden_dim-dimensional space."""
    directions = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(SEED + 100 + seed)
        d = rng.randn(hidden_dim)
        d = d / (np.linalg.norm(d) + 1e-12)
        directions.append(d)
    return directions


# ── Control direction 2: Sentiment / valence ───────────────────────────────

POSITIVE_TEXTS = [
    "I love this, it's absolutely wonderful and amazing.",
    "I'm so happy, everything is going perfectly today.",
    "This is the best thing that has ever happened to me.",
    "I feel fantastic, full of energy and excitement.",
    "What a beautiful day, I'm grateful for everything.",
    "I'm thrilled, this is incredible and makes me so proud.",
    "Everything is wonderful, I couldn't be more delighted.",
    "I feel so lucky and blessed, life is truly great.",
]

NEGATIVE_TEXTS = [
    "I hate this, it's absolutely terrible and awful.",
    "I'm so sad, everything is going horribly wrong today.",
    "This is the worst thing that has ever happened to me.",
    "I feel miserable, drained of all energy and hope.",
    "What a dreadful day, nothing is going right at all.",
    "I'm devastated, this is horrible and makes me so upset.",
    "Everything is awful, I couldn't be more disappointed.",
    "I feel so unlucky and cursed, life is truly unfair.",
]


def compute_sentiment_direction(model, tokenizer, device, peak_layer):
    """Compute positive -> negative sentiment direction at peak_layer."""
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

    neg_center = get_mean_hidden(NEGATIVE_TEXTS)
    pos_center = get_mean_hidden(POSITIVE_TEXTS)
    # Direction: negative -> positive (analogous to affective -> mentalistic)
    direction = pos_center - neg_center
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction


# ── Control direction 3: PC1 of all stimuli ────────────────────────────────

def compute_pc1_from_npz(model_short, peak_layer_idx):
    """
    Load per-stim activations and compute PC1.

    peak_layer_idx: the index into the per_stim_activations array's layer dimension.
    For Qwen2.5-7B-Instruct peak_layer=27, the NPZ has 29 entries
    (embedding + layer_0..layer_27), so peak index = 27 (matching reconstruct script).
    """
    npz_path = RSA_DIR / f"{model_short}_rsa_v2_per_stim.npz"
    print(f"  Loading per-stim NPZ: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)

    pools = list(z["pooling_names"])
    pidx = pools.index("mean_all")

    # per_stim_activations: [pool, stim, layer, hidden]
    acts = z["per_stim_activations"][pidx, :, peak_layer_idx, :]  # [712, hidden]
    acts = acts.astype(np.float64)
    print(f"  Loaded activations: {acts.shape} at layer index {peak_layer_idx}")

    # Center and compute PC1
    acts_centered = acts - acts.mean(axis=0, keepdims=True)
    # SVD for PC1 (more memory-efficient than full covariance for wide matrices)
    U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)
    pc1 = Vt[0]  # first right singular vector = first PC direction
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)

    # Report variance explained
    var_explained = S[0]**2 / np.sum(S**2)
    print(f"  PC1 variance explained: {var_explained:.3f}")
    return pc1.astype(np.float64)


def load_or_compute_pc1(model_short, peak_layer_idx):
    """Try to load precomputed PC1; if not found, compute from NPZ."""
    pc1_path = RSA_DIR / f"{model_short}_pc1_direction.npz"
    if pc1_path.exists():
        z = np.load(pc1_path)
        print(f"  Loaded precomputed PC1 from {pc1_path}")
        print(f"  PC1 variance explained: {z['variance_explained']:.3f}")
        return z["pc1"]
    return compute_pc1_from_npz(model_short, peak_layer_idx)


def precompute_and_save_pc1(model_short, peak_layer_idx):
    """Compute PC1 and save it so the GPU job doesn't need the large NPZ."""
    pc1 = compute_pc1_from_npz(model_short, peak_layer_idx)
    out_path = RSA_DIR / f"{model_short}_pc1_direction.npz"

    # Also save variance explained for documentation
    z = np.load(RSA_DIR / f"{model_short}_rsa_v2_per_stim.npz", allow_pickle=True)
    pools = list(z["pooling_names"])
    pidx = pools.index("mean_all")
    acts = z["per_stim_activations"][pidx, :, peak_layer_idx, :].astype(np.float64)
    acts_c = acts - acts.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(acts_c, full_matrices=False)
    var_exp = S[0]**2 / np.sum(S**2)

    np.savez_compressed(out_path, pc1=pc1, variance_explained=np.array(var_exp),
                        peak_layer_idx=np.array(peak_layer_idx),
                        model_short=np.array(model_short))
    print(f"  Saved PC1 direction: {out_path} ({pc1.shape[0]}-dim, var={var_exp:.3f})")
    return pc1


# ── Direction cosine similarity diagnostic ─────────────────────────────────

def direction_diagnostics(directions_dict):
    """Print pairwise cosine similarities between all directions."""
    names = list(directions_dict.keys())
    print("\n=== Direction cosine similarities ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = directions_dict[names[i]]
            b = directions_dict[names[j]]
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            print(f"  {names[i]:20s} vs {names[j]:20s}: cos = {cos:+.4f}")
    print()


# ── Generation + output ───────────────────────────────────────────────────

def run_control_condition(model, tokenizer, device, peak_layer,
                          direction, control_name, all_prompts, alphas,
                          model_short, max_new_tokens=200):
    """Generate responses for one control direction across all prompts/alphas."""
    results = []
    t0 = time.time()
    for pi, p in enumerate(all_prompts):
        print(f"\n  [{pi+1}/{len(all_prompts)}] {p['task']} / {p['id']}")
        for alpha in alphas:
            response = generate_with_steering(
                model, tokenizer, p["prompt"], direction, alpha, device,
                peak_layer, max_new_tokens=max_new_tokens)
            results.append({
                "prompt_id": p["id"],
                "task": p["task"],
                "task_type": p["type"],
                "prompt": p["prompt"],
                "alpha": alpha,
                "response": response,
                "model": model_short,
                "control": control_name,
            })
            trunc = response[:120].replace("\n", " ")
            print(f"    a={alpha:+3d}: {trunc}{'...' if len(response) > 120 else ''}")
    elapsed = time.time() - t0
    print(f"  {control_name}: {len(results)} responses in {elapsed:.0f}s")
    return results


def save_results(results, control_name, model_short):
    """Save results JSON and blind rating TSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    out_json = OUT_DIR / f"{control_name}_rating_responses.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")

    # Blind rating TSV (same format as brain-axis)
    random.seed(SEED + 42)
    shuffled = list(range(len(results)))
    random.shuffle(shuffled)

    tsv_path = OUT_DIR / f"{control_name}_rating_sheet.tsv"
    with open(tsv_path, "w") as f:
        f.write("rating_id\ttask\tprompt\tresponse\temotional_intensity_1to7\t"
                "analytical_quality_1to7\tempathy_understanding_1to7\t"
                "overall_quality_1to7\n")
        for ri, idx in enumerate(shuffled):
            r = results[idx]
            prompt_clean = r["prompt"].replace("\t", " ").replace("\n", " | ")
            resp_clean = r["response"].replace("\t", " ").replace("\n", " | ")
            f.write(f"R{ri+1:03d}\t{r['task']}\t{prompt_clean}\t{resp_clean}"
                    f"\t\t\t\t\n")
    print(f"Rating sheet: {tsv_path}")


def save_combined_ranking(brain_results, random_results_all, model_short):
    """
    Combined ranking sheet: for each prompt, show 4 responses --
    brain alpha=-10, brain alpha=+10, random(seed0) alpha=-10, random(seed0) alpha=+10.
    Shuffled within each prompt, labeled A-D. Rater ranks most emotional to most analytical.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load brain-axis results if not passed
    if brain_results is None:
        brain_path = OUT_DIR / f"{model_short}_rating_responses.json"
        if not brain_path.exists():
            print(f"WARNING: Brain-axis results not found at {brain_path}, "
                  f"skipping combined ranking.")
            return
        brain_results = json.load(open(brain_path))

    # Use only the first random seed's results for the comparison
    random_results = random_results_all[0] if random_results_all else []

    # Index by prompt_id for quick lookup
    def index_by(results, alpha_val):
        return {r["prompt_id"]: r for r in results if r["alpha"] == alpha_val}

    brain_neg = index_by(brain_results, -10)
    brain_pos = index_by(brain_results, 10)
    rand_neg = index_by(random_results, -10)
    rand_pos = index_by(random_results, 10)

    # Get all prompt_ids that appear in all four
    prompt_ids = sorted(set(brain_neg.keys()) & set(brain_pos.keys())
                        & set(rand_neg.keys()) & set(rand_pos.keys()))

    tsv_path = OUT_DIR / f"combined_brain_vs_random_ranking.tsv"
    key_path = OUT_DIR / f"combined_brain_vs_random_key.tsv"

    rng = random.Random(SEED + 999)

    with open(tsv_path, "w") as f, open(key_path, "w") as fk:
        # Header
        f.write("question\ttask\tprompt\tresponse_A\tresponse_B\t"
                "response_C\tresponse_D\t"
                "rank_most_emotional_to_most_analytical(ABCD)\n")
        fk.write("question\tlabel_A\tlabel_B\tlabel_C\tlabel_D\n")

        for qi, pid in enumerate(prompt_ids):
            entries = [
                ("brain_a-10", brain_neg[pid]),
                ("brain_a+10", brain_pos[pid]),
                ("random_a-10", rand_neg[pid]),
                ("random_a+10", rand_pos[pid]),
            ]
            rng.shuffle(entries)
            labels = [e[0] for e in entries]
            resps = [e[1]["response"].replace("\t", " ").replace("\n", " | ")
                     for e in entries]
            prompt_clean = entries[0][1]["prompt"].replace("\t", " ").replace("\n", " | ")
            task = entries[0][1]["task"]

            f.write(f"Q{qi+1:02d}\t{task}\t{prompt_clean}\t"
                    f"{resps[0]}\t{resps[1]}\t{resps[2]}\t{resps[3]}\t\n")
            fk.write(f"Q{qi+1:02d}\t{labels[0]}\t{labels[1]}\t"
                     f"{labels[2]}\t{labels[3]}\n")

    print(f"Combined ranking: {tsv_path}")
    print(f"Answer key:       {key_path}")


def quality_check(results, control_name, alphas):
    """Print quality summary for degenerate response detection."""
    print(f"\n--- Quality check: {control_name} ---")
    for alpha in alphas:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        if not alpha_results:
            continue
        lengths = [len(r["response"]) for r in alpha_results]
        empty = sum(1 for l in lengths if l < 10)
        short = sum(1 for l in lengths if l < 50)
        print(f"  a={alpha:+3d}: n={len(alpha_results)}, "
              f"mean_len={np.mean(lengths):.0f}, "
              f"empty(<10)={empty}, short(<50)={short}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Control-direction steering for brain-axis specificity test")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model (required unless --precompute_pc1_only)")
    parser.add_argument("--model_short", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--peak_layer", type=int, default=27)
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--controls", default="random,sentiment,pc1",
                        help="Comma-separated list of controls to run "
                             "(random, sentiment, pc1)")
    parser.add_argument("--precompute_pc1_only", action="store_true",
                        help="Only precompute and save PC1 direction (no GPU needed)")
    parser.add_argument("--skip_combined", action="store_true",
                        help="Skip generating combined ranking sheet")
    args = parser.parse_args()

    # ── Precompute PC1 mode (CPU, no model needed) ─────────────────────────
    if args.precompute_pc1_only:
        print("=== Precomputing PC1 direction (CPU only) ===")
        precompute_and_save_pc1(args.model_short, args.peak_layer)
        return

    # ── Full run: needs GPU + model ────────────────────────────────────────
    if args.model_path is None:
        parser.error("--model_path is required for full run")

    alphas = [int(a) for a in args.alphas.split(",")]
    controls_to_run = [c.strip() for c in args.controls.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts (same 30 as brain-axis experiment)
    moral_prompts = load_moral_prompts(10)
    empathy_prompts = load_empathy_prompts(10)
    tom_prompts = load_tom_prompts(10)
    all_prompts = moral_prompts + empathy_prompts + tom_prompts
    print(f"Prompts: {len(moral_prompts)} moral + {len(empathy_prompts)} empathy "
          f"+ {len(tom_prompts)} ToM = {len(all_prompts)}")
    print(f"Alphas: {alphas}")
    print(f"Controls: {controls_to_run}")

    # Load model
    print(f"\nLoading {args.model_short} from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    hidden_dim = model.config.hidden_size
    print(f"Model loaded. hidden_dim={hidden_dim}, device={device}")

    # Compute brain-axis direction (for diagnostics, not used for steering here)
    print("\nComputing brain-axis direction (for cosine similarity diagnostics)...")
    brain_dir = compute_boundary_direction(model, tokenizer, device, args.peak_layer)

    # ── Build all control directions ───────────────────────────────────────
    all_directions = {"brain_axis": brain_dir}
    all_random_results = []

    if "random" in controls_to_run:
        print("\n=== Control 1: Random directions ===")
        random_dirs = compute_random_directions(hidden_dim)
        for si, rd in enumerate(random_dirs):
            all_directions[f"random_seed{si}"] = rd

    if "sentiment" in controls_to_run:
        print("\n=== Control 2: Sentiment direction ===")
        sentiment_dir = compute_sentiment_direction(
            model, tokenizer, device, args.peak_layer)
        all_directions["sentiment"] = sentiment_dir

    if "pc1" in controls_to_run:
        print("\n=== Control 3: PC1 of all stimuli ===")
        pc1_dir = load_or_compute_pc1(args.model_short, args.peak_layer)
        all_directions["pc1"] = pc1_dir

    # Print direction diagnostics
    direction_diagnostics(all_directions)

    # ── Run each control condition ─────────────────────────────────────────

    if "random" in controls_to_run:
        print("\n" + "=" * 60)
        print("GENERATING: Random control (5 seeds)")
        print("=" * 60)
        for si in range(N_RANDOM_SEEDS):
            seed_name = f"random_seed{si}"
            print(f"\n--- Random seed {si} ---")
            results_si = run_control_condition(
                model, tokenizer, device, args.peak_layer,
                all_directions[seed_name], seed_name,
                all_prompts, alphas, args.model_short, args.max_new_tokens)
            all_random_results.append(results_si)
            quality_check(results_si, seed_name, alphas)

        # Aggregate: save all 5 seeds in one file, plus a median-effect summary
        all_random_flat = []
        for si, results_si in enumerate(all_random_results):
            for r in results_si:
                r["random_seed"] = si
            all_random_flat.extend(results_si)

        save_results(all_random_flat, "random_all_seeds", args.model_short)

        # Also save seed 0 as the canonical "random" for the ranking sheet
        save_results(all_random_results[0], "random", args.model_short)

    if "sentiment" in controls_to_run:
        print("\n" + "=" * 60)
        print("GENERATING: Sentiment control")
        print("=" * 60)
        sentiment_results = run_control_condition(
            model, tokenizer, device, args.peak_layer,
            all_directions["sentiment"], "sentiment",
            all_prompts, alphas, args.model_short, args.max_new_tokens)
        save_results(sentiment_results, "sentiment", args.model_short)
        quality_check(sentiment_results, "sentiment", alphas)

    if "pc1" in controls_to_run:
        print("\n" + "=" * 60)
        print("GENERATING: PC1 control")
        print("=" * 60)
        pc1_results = run_control_condition(
            model, tokenizer, device, args.peak_layer,
            all_directions["pc1"], "pc1",
            all_prompts, alphas, args.model_short, args.max_new_tokens)
        save_results(pc1_results, "pc1", args.model_short)
        quality_check(pc1_results, "pc1", alphas)

    # ── Combined brain-vs-random ranking sheet ─────────────────────────────
    if not args.skip_combined and "random" in controls_to_run:
        print("\n" + "=" * 60)
        print("GENERATING: Combined brain-vs-random ranking sheet")
        print("=" * 60)
        # Load brain-axis results from the existing file
        brain_path = OUT_DIR / f"{args.model_short}_rating_responses.json"
        brain_results = None
        if brain_path.exists():
            brain_results = json.load(open(brain_path))
        save_combined_ranking(brain_results, all_random_results, args.model_short)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0
    if "random" in controls_to_run:
        n = len(all_random_flat)
        total += n
        print(f"  random: {n} responses ({N_RANDOM_SEEDS} seeds x "
              f"{len(all_prompts)} prompts x {len(alphas)} alphas)")
    if "sentiment" in controls_to_run:
        n = len(all_prompts) * len(alphas)
        total += n
        print(f"  sentiment: {n} responses")
    if "pc1" in controls_to_run:
        n = len(all_prompts) * len(alphas)
        total += n
        print(f"  pc1: {n} responses")
    print(f"  TOTAL: {total} responses")
    print(f"  Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
