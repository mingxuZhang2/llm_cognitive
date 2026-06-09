#!/usr/bin/env python3
"""
A: Narrative-frame steering vs VA steering.

If narrative frame is the model's real organizing principle (not VA), then
steering along narrative-frame axes should produce LARGER behavioral effects
than steering along VA axes. This directly challenges Sun et al. 2026 and
Anthropic 2026's VA-based approach.

Method:
1. Compute narrative frame direction: centroid(moral) - centroid(threat)
2. Compute valence direction: regression of valence scores on activations
3. Compute arousal direction: regression of arousal scores on activations
4. Steer along each direction, measure JSD of next-token predictions
5. Compare effect sizes

GPU needed.
"""
from __future__ import annotations
import json, csv, re, itertools
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr, wilcoxon
from scipy.spatial.distance import jensenshannon

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "mechanistic"
OUT_DIR.mkdir(exist_ok=True)

# Stimuli for computing steering directions
MORAL_STIMS = [
    "The politician was caught embezzling money meant for disaster relief.",
    "A pharmaceutical company covered up dangerous side effects to protect profits.",
    "The judge accepted bribes to reduce sentences for wealthy defendants.",
    "A corporation dumped toxic waste into the river that supplied drinking water.",
    "The teacher was found fabricating grades in exchange for personal favors.",
]
THREAT_STIMS = [
    "The bridge started swaying violently as the storm intensified overhead.",
    "She heard footsteps approaching rapidly in the dark empty parking garage.",
    "The pilot announced they were making an emergency landing due to engine failure.",
    "He noticed the car behind him had been following for the last thirty minutes.",
    "The evacuation alarm blared as smoke filled the stairwell of the building.",
]
INTERPERSONAL_STIMS = [
    "After twenty years of marriage, he found messages from someone else on her phone.",
    "Her best friend shared her deepest secret with everyone at the party.",
    "His brother borrowed his life savings and gambled it all away without telling him.",
    "She overheard her parents saying they wished she was more like her sister.",
    "His mentor took credit for the breakthrough idea they had discussed in private.",
]

# Test prompts (neutral situations where steering should shift the response)
TEST_PROMPTS = [
    "My neighbor has been acting strangely lately. What do you think about this?",
    "I received an unexpected letter from someone I haven't heard from in years. What should I make of this?",
    "There's been a change in management at my company. How should I approach this?",
    "I noticed something unusual in my bank statement. What would you advise?",
    "A friend asked me to keep a secret that involves other people. What should I do?",
    "I found out something about a colleague that surprised me. How should I handle this?",
    "My child came home from school and seemed different. What should I think about this?",
    "I was offered an opportunity that seems too good to be true. What's your take?",
    "Someone left an anonymous note on my car. What do you make of this?",
    "A family member made a decision that affects everyone without consulting us. What should I do?",
]


def get_mean_activation(model, tokenizer, texts, device, layer_frac=0.75):
    """Get mean activation from last 25% of layers for a set of texts."""
    all_acts = []
    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * layer_frac)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[target_layer][0].float()
        mask = inputs["attention_mask"][0].bool()
        act = hidden[mask].mean(dim=0)
        all_acts.append(act)
    return torch.stack(all_acts).mean(dim=0)


def get_next_token_dist(model, tokenizer, text, device, hooks=None, top_k=500):
    """Get next-token probability distribution, optionally with steering hooks."""
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1).cpu()
    topk_probs, topk_ids = torch.topk(probs, top_k)
    dist = torch.zeros(probs.shape[0])
    dist[topk_ids] = topk_probs
    return (dist / dist.sum()).numpy()


def steer_and_measure(model, tokenizer, prompts, direction, alphas, device):
    """Steer along direction at multiple alphas, measure JSD from baseline."""
    n_layers = model.config.num_hidden_layers
    target_layers = list(range(n_layers * 3 // 4, n_layers))

    # Baseline (alpha=0)
    baselines = []
    for prompt in prompts:
        baselines.append(get_next_token_dist(model, tokenizer, prompt, device))

    results_per_alpha = {}
    for alpha in alphas:
        hooks = []

        def make_hook(a):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h = h + a * direction.to(h.device, h.dtype).unsqueeze(0).unsqueeze(0)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return hook_fn

        for li in target_layers:
            h = model.model.layers[li].register_forward_hook(make_hook(alpha))
            hooks.append(h)

        jsds = []
        for pi, prompt in enumerate(prompts):
            steered = get_next_token_dist(model, tokenizer, prompt, device)
            j = float(jensenshannon(baselines[pi], steered))
            if not np.isnan(j):
                jsds.append(j)

        for h in hooks:
            h.remove()

        results_per_alpha[alpha] = {
            "mean_jsd": float(np.mean(jsds)) if jsds else 0,
            "std_jsd": float(np.std(jsds)) if jsds else 0,
        }
        print(f"    alpha={alpha:+.1f}: mean JSD={np.mean(jsds):.5f}")

    return results_per_alpha


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Narrative vs VA Steering ({model_short}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    # Step 1: Compute steering directions
    print("\n--- Computing steering directions ---")

    moral_center = get_mean_activation(model, tokenizer, MORAL_STIMS, device)
    threat_center = get_mean_activation(model, tokenizer, THREAT_STIMS, device)
    interp_center = get_mean_activation(model, tokenizer, INTERPERSONAL_STIMS, device)

    # Narrative frame direction: moral → threat
    narr_dir = threat_center - moral_center
    narr_dir = narr_dir / narr_dir.norm()

    # For VA: use a simple proxy — positive vs negative valence stimuli
    pos_stims = [
        "The community came together to rebuild after the disaster, everyone helping each other.",
        "She received the scholarship she had been working toward for three years.",
        "The children laughed and played in the park on a beautiful sunny afternoon.",
        "He finally reconciled with his estranged brother after years of silence.",
        "The team celebrated their victory, hugging and cheering with pure joy.",
    ]
    neg_stims = [
        "The factory closed without warning, leaving hundreds of families without income.",
        "She sat alone in the hospital waiting room, dreading the doctor's next words.",
        "The old man watched helplessly as his home was foreclosed and strangers moved his things out.",
        "He read the rejection letter for the tenth time, each word cutting deeper.",
        "The funeral was silent except for the sound of rain on the coffin.",
    ]

    pos_center = get_mean_activation(model, tokenizer, pos_stims, device)
    neg_center = get_mean_activation(model, tokenizer, neg_stims, device)

    # Valence direction: negative → positive
    valence_dir = pos_center - neg_center
    valence_dir = valence_dir / valence_dir.norm()

    # High vs low arousal
    high_arousal_stims = [
        "The explosion shook the building and everyone scrambled for the exits.",
        "She sprinted across the field, heart pounding, lungs burning, with seconds to spare.",
        "The crowd erupted in a deafening roar as the final goal was scored.",
        "He slammed the brakes as the truck appeared from nowhere, tires screeching.",
        "The rollercoaster plunged straight down and she screamed at the top of her lungs.",
    ]
    low_arousal_stims = [
        "She sat quietly by the window, watching the rain trickle down the glass.",
        "The old cat slept peacefully on the warm blanket, barely stirring.",
        "He slowly turned the pages of the worn book, lost in gentle memories.",
        "The garden was still in the early morning, dew glistening on each leaf.",
        "She lay in the hammock, listening to the distant sound of waves.",
    ]

    high_center = get_mean_activation(model, tokenizer, high_arousal_stims, device)
    low_center = get_mean_activation(model, tokenizer, low_arousal_stims, device)

    arousal_dir = high_center - low_center
    arousal_dir = arousal_dir / arousal_dir.norm()

    # Random direction (control)
    rng = np.random.default_rng(42)
    random_dir = torch.tensor(rng.standard_normal(narr_dir.shape[0]), dtype=torch.float32)
    random_dir = random_dir / random_dir.norm()

    # Cosine similarities between directions
    cos = lambda a, b: float(torch.dot(a.cpu(), b.cpu()))
    print(f"\n  Direction cosines:")
    print(f"    narrative vs valence: {cos(narr_dir, valence_dir):.3f}")
    print(f"    narrative vs arousal: {cos(narr_dir, arousal_dir):.3f}")
    print(f"    valence vs arousal:   {cos(valence_dir, arousal_dir):.3f}")
    print(f"    narrative vs random:  {cos(narr_dir, random_dir.to(narr_dir.device)):.3f}")

    # Step 2: Steer and measure
    print("\n--- Steering experiments ---")
    alphas = [-3, -1, 1, 3]

    directions = {
        "narrative_frame": narr_dir,
        "valence": valence_dir,
        "arousal": arousal_dir,
        "random": random_dir.to(narr_dir.device),
    }

    all_results = {}
    for dir_name, direction in directions.items():
        print(f"\n  [{dir_name}]")
        res = steer_and_measure(model, tokenizer, TEST_PROMPTS, direction, alphas, device)
        all_results[dir_name] = res

    # Step 3: Compare effect sizes
    print(f"\n{'='*60}")
    print("COMPARISON: Which direction produces larger behavioral change?")
    print(f"{'='*60}")

    print(f"\n  {'Direction':>20s} | {'Mean JSD (|α|=3)':>16s} | {'Mean JSD (|α|=1)':>16s}")
    print(f"  {'-'*58}")

    for dir_name in directions:
        jsd_3 = np.mean([all_results[dir_name][a]["mean_jsd"] for a in [3, -3]])
        jsd_1 = np.mean([all_results[dir_name][a]["mean_jsd"] for a in [1, -1]])
        print(f"  {dir_name:>20s} | {jsd_3:16.5f} | {jsd_1:16.5f}")

    # Save
    results = {
        "model": model_short,
        "n_test_prompts": len(TEST_PROMPTS),
        "alphas": alphas,
        "direction_cosines": {
            "narrative_valence": cos(narr_dir, valence_dir),
            "narrative_arousal": cos(narr_dir, arousal_dir),
            "valence_arousal": cos(valence_dir, arousal_dir),
        },
        "steering_results": {k: {str(a): v for a, v in res.items()} for k, res in all_results.items()},
    }
    json.dump(results, open(OUT_DIR / f"narrative_vs_va_steering_{model_short}.json", "w"), indent=2)
    print(f"\nSaved results")
    print("DONE")


if __name__ == "__main__":
    main()
