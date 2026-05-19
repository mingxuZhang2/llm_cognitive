"""
Phase 0: Prepare stimulus dataset (15 categories, all from public datasets).

No custom templates. Each category sourced from established benchmarks/datasets.
Output: JSONL file with fields {text, category, source, idx}.
"""

import json
import os
import random
from pathlib import Path
from datasets import load_dataset


CATEGORIES = {
    # --- Math & Reasoning (4) ---
    "math_word_problems": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "text_field": "question",
        "n": 400,
    },
    "formal_math": {
        "dataset": "hendrycks/competition_math",
        "config": None,
        "split": "train",
        "text_field": "problem",
        "n": 400,
    },
    "logical_reasoning": {
        "dataset": "lucasmccabe/logiqa",
        "config": None,
        "split": "train",
        "text_field": lambda x: x["context"] + "\nQuestion: " + x["query"],
        "n": 400,
    },
    "commonsense_reasoning": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "train",
        "text_field": "question",
        "n": 400,
    },

    # --- Code (2) ---
    "code_generation": {
        "dataset": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "text_field": "prompt",
        "n": 164,  # full HumanEval
    },
    "code_problems": {
        "dataset": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "train",
        "text_field": "text",
        "n": 400,
    },

    # --- Language (3) ---
    "translation": {
        "dataset": "facebook/flores",
        "config": "eng_Latn-zho_Hans",
        "split": "devtest",
        "text_field": lambda x: "Translate to Chinese: " + x["sentence_eng_Latn"],
        "n": 400,
    },
    "summarization": {
        "dataset": "EdinburghNLP/xsum",
        "config": None,
        "split": "train",
        "text_field": lambda x: "Summarize:\n" + x["document"][:1500],
        "n": 400,
    },
    "text_completion": {
        "dataset": "Rowan/hellaswag",
        "config": None,
        "split": "train",
        "text_field": lambda x: x["activity_label"] + ": " + x["ctx"],
        "n": 400,
    },

    # --- Knowledge (3) ---
    "science_knowledge": {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
        "text_field": lambda x: x["question"] + "\n" + "\n".join(
            f"({chr(65+i)}) {c}" for i, c in enumerate(x["choices"])
        ),
        "filter": lambda x: x.get("subject", "") in [
            "physics", "chemistry", "biology", "computer_science",
            "mathematics", "astronomy", "machine_learning",
            "college_physics", "college_chemistry", "college_biology",
        ],
        "n": 400,
    },
    "humanities_knowledge": {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
        "text_field": lambda x: x["question"] + "\n" + "\n".join(
            f"({chr(65+i)}) {c}" for i, c in enumerate(x["choices"])
        ),
        "filter": lambda x: x.get("subject", "") in [
            "philosophy", "world_history", "us_history", "world_religions",
            "prehistory", "high_school_european_history", "moral_scenarios",
        ],
        "n": 400,
    },
    "factual_qa": {
        "dataset": "mandarjoshi/trivia_qa",
        "config": "rc.nocontext",
        "split": "train",
        "text_field": "question",
        "n": 400,
    },

    # --- Ethics & Social (2) ---
    "ethical_judgment": {
        "dataset": "hendrycks/ethics",
        "config": "commonsense",
        "split": "train",
        "text_field": lambda x: "Is this ethical? " + x["input"],
        "n": 400,
    },
    "truthfulness": {
        "dataset": "truthful_qa",
        "config": "generation",
        "split": "validation",
        "text_field": "question",
        "n": 400,
    },
}


def load_category(name, cfg, seed=42):
    """Load samples for a single category from HuggingFace."""
    rng = random.Random(seed)
    print(f"  Loading {name} from {cfg['dataset']}...")

    try:
        if cfg["config"]:
            ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"], trust_remote_code=True)
        else:
            ds = load_dataset(cfg["dataset"], split=cfg["split"], trust_remote_code=True)
    except Exception as e:
        print(f"    WARNING: Failed to load {cfg['dataset']}: {e}")
        return []

    items = list(ds)

    if "filter" in cfg:
        items = [x for x in items if cfg["filter"](x)]

    if len(items) > cfg["n"]:
        items = rng.sample(items, cfg["n"])

    samples = []
    for item in items:
        tf = cfg["text_field"]
        if callable(tf):
            text = tf(item)
        else:
            text = item[tf]

        if text and len(text.strip()) > 10:
            samples.append({
                "text": text.strip()[:2000],
                "category": name,
                "source": cfg["dataset"],
            })

    print(f"    Got {len(samples)} samples")
    return samples


def prepare_all_stimuli(output_path, seed=42):
    """Prepare the full stimulus dataset from public datasets only."""
    all_samples = []

    print(f"Preparing stimuli ({len(CATEGORIES)} categories)...\n")
    for name, cfg in CATEGORIES.items():
        samples = load_category(name, cfg, seed)
        all_samples.extend(samples)

    random.Random(seed).shuffle(all_samples)

    for i, s in enumerate(all_samples):
        s["idx"] = i

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Category counts
    counts = {}
    for s in all_samples:
        counts[s["category"]] = counts.get(s["category"], 0) + 1

    index_path = output_path.replace(".jsonl", "_index.json")
    with open(index_path, "w") as f:
        json.dump(counts, f, indent=2)

    print(f"\nTotal: {len(all_samples)} samples across {len(counts)} categories")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:30s} {n:>4d}")
    print(f"\nSaved to: {output_path}")

    return all_samples


def prepare_pilot_stimuli(output_path, n_per_category=50, seed=42):
    """Prepare a small pilot stimulus set for code testing."""
    all_samples = []

    print(f"Preparing PILOT stimuli ({n_per_category}/category)...\n")
    for name, cfg in CATEGORIES.items():
        pilot_cfg = dict(cfg)
        pilot_cfg["n"] = n_per_category
        samples = load_category(name, pilot_cfg, seed)
        all_samples.extend(samples)

    random.Random(seed).shuffle(all_samples)
    for i, s in enumerate(all_samples):
        s["idx"] = i

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Pilot set: {len(all_samples)} samples")
    print(f"Saved to: {output_path}")
    return all_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pilot", action="store_true", help="Generate small pilot set instead")
    parser.add_argument("--n_pilot", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.pilot:
        prepare_pilot_stimuli(args.output, args.n_pilot, args.seed)
    else:
        prepare_all_stimuli(args.output, args.seed)
