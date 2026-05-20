"""
Prepare stimuli with gold answers for downstream accuracy evaluation.

Loads from HuggingFace, extracts gold answers per source dataset, and
produces stimuli files with fields: {text, category, source, idx, answer, answer_type}

Answer types:
  - exact_match:     model output must match answer string (GSM8K numbers, TriviaQA)
  - multiple_choice: model must select correct letter A/B/C/D (ARC, MMLU)
  - completion:      correct continuation text (HellaSwag)
  - generation:      reference text for log-prob or ROUGE scoring (HumanEval, TruthfulQA)

Creates:
  - stimuli_with_answers_full.jsonl      (all samples, unbalanced)
  - stimuli_with_answers_medium.jsonl    (50/category, balanced)
  - stimuli_with_answers_balanced.jsonl  (min_count/category, balanced)
  - stimuli_with_answers_balanced_discovery.jsonl  (50% of balanced)
  - stimuli_with_answers_balanced_validation.jsonl (50% of balanced)
  - stimuli_with_answers_medium_discovery.jsonl    (50% of medium)
  - stimuli_with_answers_medium_validation.jsonl   (50% of medium)

Uses the same 8 simplified categories as the existing pipeline:
  math, code, reasoning, language, science, humanities, factual_qa, ethics
"""

import json
import random
import argparse
import re
import gzip
import io
from pathlib import Path
from collections import Counter, defaultdict

import requests
from datasets import load_dataset


# ---------------------------------------------------------------------------
# MMLU subject lists (from original stimulus_preparation.py)
# ---------------------------------------------------------------------------
MMLU_STEM_SUBJECTS = [
    "physics", "chemistry", "biology", "computer_science",
    "mathematics", "astronomy", "machine_learning",
    "college_physics", "college_chemistry", "college_biology",
]

MMLU_HUMANITIES_SUBJECTS = [
    "philosophy", "world_history", "us_history", "world_religions",
    "prehistory", "high_school_european_history", "moral_scenarios",
]


# ---------------------------------------------------------------------------
# Per-source loading + answer extraction
# ---------------------------------------------------------------------------

def load_gsm8k(n, rng):
    """math: GSM8K -- answer is the number after ####."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = list(ds)
    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        text = item["question"].strip()
        # Extract numeric answer after "####"
        raw_answer = item["answer"]
        match = re.search(r"####\s*(.+)", raw_answer)
        if match:
            answer = match.group(1).strip().replace(",", "")
        else:
            # Fallback: take last number in the answer
            nums = re.findall(r"[\d,]+\.?\d*", raw_answer)
            answer = nums[-1].replace(",", "") if nums else raw_answer.strip()

        if text and len(text) > 10:
            samples.append({
                "text": text[:2000],
                "category": "math",
                "source": "gsm8k",
                "answer": answer,
                "answer_type": "exact_match",
            })
    return samples


def _download_humaneval_from_github():
    """Download HumanEval dataset from the official GitHub repository.

    The openai/openai_humaneval HuggingFace dataset uses a legacy loading
    script that is no longer supported in datasets>=4.0. We fall back to
    downloading the gzipped JSONL directly from GitHub.
    """
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    print(f"    Downloading HumanEval from GitHub: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw = gzip.decompress(resp.content).decode("utf-8")
    items = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
    print(f"    Downloaded {len(items)} HumanEval problems")
    return items


def load_humaneval(n, rng):
    """code: HumanEval -- answer is canonical_solution."""
    # Try HuggingFace first, fall back to GitHub download
    items = None
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
        items = list(ds)
        print("    Loaded from HuggingFace")
    except Exception:
        pass

    if items is None:
        try:
            items = _download_humaneval_from_github()
        except Exception as e:
            print(f"    WARNING: GitHub download failed: {e}")
            print("    Falling back to MBPP as code source")
            return _load_mbpp_fallback(n, rng)

    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        text = item["prompt"].strip()
        answer = item["canonical_solution"].strip()

        if text and len(text) > 10:
            samples.append({
                "text": text[:2000],
                "category": "code",
                "source": "humaneval",
                "answer": answer,
                "answer_type": "generation",
            })
    return samples


def _load_mbpp_fallback(n, rng):
    """Fallback: use MBPP if HumanEval is unavailable."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    items = list(ds)
    if len(items) > n:
        items = rng.sample(items, n)
    samples = []
    for item in items:
        text = item.get("text", item.get("prompt", "")).strip()
        answer = item.get("code", "").strip()
        if text and len(text) > 10 and answer:
            samples.append({
                "text": text[:2000],
                "category": "code",
                "source": "mbpp",
                "answer": answer[:2000],
                "answer_type": "generation",
            })
    return samples


def load_arc_challenge(n, rng):
    """reasoning: ARC-Challenge -- MC with correct answer letter."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    items = list(ds)
    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        question = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]

        # Format with lettered choices
        choice_str = "\n".join(
            f"({lbl}) {txt}" for lbl, txt in zip(labels, texts)
        )
        text = f"{question}\n{choice_str}"

        answer = item["answerKey"].strip()

        if text and len(text) > 10:
            samples.append({
                "text": text[:2000],
                "category": "reasoning",
                "source": "arc",
                "answer": answer,
                "answer_type": "multiple_choice",
            })
    return samples


def load_hellaswag(n, rng):
    """language: HellaSwag -- answer is the correct ending."""
    ds = load_dataset("Rowan/hellaswag", split="train")
    items = list(ds)
    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        # Use the same text format as original pipeline
        activity = item.get("activity_label", "")
        ctx = item.get("ctx", "")
        text = f"{activity}: {ctx}".strip() if activity else ctx.strip()

        label = int(item["label"])
        endings = item["endings"]
        answer = endings[label] if label < len(endings) else endings[0]

        if text and len(text) > 10:
            samples.append({
                "text": text[:2000],
                "category": "language",
                "source": "hellaswag",
                "answer": answer.strip(),
                "answer_type": "completion",
            })
    return samples


def _load_mmlu_all():
    """Load and cache the full MMLU test set (shared between STEM and humanities)."""
    if not hasattr(_load_mmlu_all, "_cache"):
        _load_mmlu_all._cache = load_dataset("cais/mmlu", "all", split="test")
    return _load_mmlu_all._cache


def load_mmlu_stem(n, rng):
    """science: MMLU-STEM -- MC with correct letter."""
    ds = _load_mmlu_all()
    items = [x for x in ds if x.get("subject", "") in MMLU_STEM_SUBJECTS]
    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        question = item["question"].strip()
        choices = item["choices"]
        choice_str = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )
        text = f"{question}\n{choice_str}"

        # answer field is integer index 0-3 -> A-D
        answer_idx = int(item["answer"])
        answer = chr(65 + answer_idx)

        if text and len(text) > 10:
            samples.append({
                "text": text[:2000],
                "category": "science",
                "source": "mmlu_stem",
                "answer": answer,
                "answer_type": "multiple_choice",
            })
    return samples


def load_mmlu_humanities(n, rng):
    """humanities: MMLU-Humanities -- MC with correct letter."""
    ds = _load_mmlu_all()
    items = [x for x in ds if x.get("subject", "") in MMLU_HUMANITIES_SUBJECTS]
    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        question = item["question"].strip()
        choices = item["choices"]
        choice_str = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )
        text = f"{question}\n{choice_str}"

        answer_idx = int(item["answer"])
        answer = chr(65 + answer_idx)

        if text and len(text) > 10:
            samples.append({
                "text": text[:2000],
                "category": "humanities",
                "source": "mmlu_humanities",
                "answer": answer,
                "answer_type": "multiple_choice",
            })
    return samples


def load_triviaqa(n, rng):
    """factual_qa: TriviaQA -- answer is the canonical value or first alias."""
    # Use the cached dataset name (trivia_qa, not mandarjoshi/trivia_qa)
    try:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    except Exception:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    items = list(ds)
    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        text = item["question"].strip()

        ans_obj = item["answer"]
        # Primary: value field; also store aliases for fuzzy matching
        answer = ans_obj.get("value", "")
        if not answer:
            aliases = ans_obj.get("aliases", [])
            answer = aliases[0] if aliases else ""
        # Store all aliases for flexible matching later
        aliases = ans_obj.get("aliases", [])

        if text and len(text) > 10 and answer:
            entry = {
                "text": text[:2000],
                "category": "factual_qa",
                "source": "trivia_qa",
                "answer": answer.strip(),
                "answer_type": "exact_match",
            }
            if aliases:
                entry["answer_aliases"] = [a.strip() for a in aliases if a.strip()]
            samples.append(entry)
    return samples


def load_truthfulqa(n, rng):
    """ethics: TruthfulQA -- answer is best_answer."""
    # Use the cached dataset name (truthful_qa, not truthfulqa/truthful_qa)
    try:
        ds = load_dataset("truthful_qa", "generation", split="validation")
    except Exception:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    items = list(ds)
    if len(items) > n:
        items = rng.sample(items, n)

    samples = []
    for item in items:
        text = item["question"].strip()
        # best_answer is the gold reference
        answer = item.get("best_answer", "")
        if not answer:
            # Fallback to correct_answers[0]
            correct = item.get("correct_answers", [])
            answer = correct[0] if correct else ""

        if text and len(text) > 10 and answer:
            samples.append({
                "text": text[:2000],
                "category": "ethics",
                "source": "truthful_qa",
                "answer": answer.strip(),
                "answer_type": "generation",
            })
    return samples


# ---------------------------------------------------------------------------
# Loader registry (category -> loader function)
# ---------------------------------------------------------------------------
LOADERS = {
    "math":       load_gsm8k,
    "code":       load_humaneval,
    "reasoning":  load_arc_challenge,
    "language":   load_hellaswag,
    "science":    load_mmlu_stem,
    "humanities": load_mmlu_humanities,
    "factual_qa": load_triviaqa,
    "ethics":     load_truthfulqa,
}

# Max samples to request per category for full set
MAX_N = {
    "math": 400,
    "code": 164,       # HumanEval only has 164 problems
    "reasoning": 400,
    "language": 400,
    "science": 400,
    "humanities": 400,
    "factual_qa": 400,
    "ethics": 400,      # TruthfulQA validation ~817
}


# ---------------------------------------------------------------------------
# Balancing and splitting utilities
# ---------------------------------------------------------------------------

def balance(samples, n_per_cat, seed=42):
    """Subsample each category to n_per_cat, shuffle, re-index."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for s in samples:
        by_cat[s["category"]].append(s)

    balanced = []
    for cat in sorted(by_cat.keys()):
        pool = by_cat[cat]
        if len(pool) <= n_per_cat:
            balanced.extend(pool)
        else:
            balanced.extend(rng.sample(pool, n_per_cat))

    rng.shuffle(balanced)
    for i, s in enumerate(balanced):
        s["idx"] = i
    return balanced


def discovery_validation_split(samples, seed=42):
    """Split balanced samples 50/50 into discovery and validation sets.

    The split is done per-category to ensure each set has equal
    representation of all 8 categories.
    """
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for s in samples:
        by_cat[s["category"]].append(s)

    discovery, validation = [], []
    for cat in sorted(by_cat.keys()):
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        mid = len(pool) // 2
        discovery.extend(pool[:mid])
        validation.extend(pool[mid:])

    # Shuffle each set and re-index
    rng.shuffle(discovery)
    rng.shuffle(validation)
    for i, s in enumerate(discovery):
        s["idx"] = i
    for i, s in enumerate(validation):
        s["idx"] = i

    return discovery, validation


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_jsonl(samples, path):
    """Write samples to JSONL and companion index JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    counts = Counter(s["category"] for s in samples)
    index_path = path.replace(".jsonl", "_index.json")
    with open(index_path, "w") as f:
        json.dump(dict(sorted(counts.items())), f, indent=2)

    return counts


def print_counts(counts, label=""):
    """Pretty-print category counts."""
    total = sum(counts.values())
    print(f"  {label} ({total} total):")
    for cat, n in sorted(counts.items()):
        print(f"    {cat:20s} {n:>4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare stimuli with gold answers for accuracy evaluation."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write output files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--medium_n", type=int, default=50,
                        help="Samples per category for medium set (default: 50)")
    args = parser.parse_args()

    seed = args.seed
    rng = random.Random(seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all categories ----
    all_samples = []
    for cat in sorted(LOADERS.keys()):
        n = MAX_N.get(cat, 400)
        print(f"\n[{cat}] Loading up to {n} samples...")
        loader = LOADERS[cat]
        samples = loader(n, rng)
        print(f"  -> Got {len(samples)} samples with answers")
        all_samples.extend(samples)

    # Shuffle and assign global indices
    rng.shuffle(all_samples)
    for i, s in enumerate(all_samples):
        s["idx"] = i

    # ---- Full (unbalanced) ----
    full_path = str(output_dir / "stimuli_with_answers_full.jsonl")
    counts_full = save_jsonl(all_samples, full_path)
    print(f"\n{'='*60}")
    print(f"FULL set saved to: {full_path}")
    print_counts(counts_full, "Full")

    # ---- Balanced (min_count/category) ----
    min_n = min(counts_full.values())
    min_cat = min(counts_full, key=counts_full.get)
    print(f"\nSmallest category: {min_cat} = {min_n}")

    balanced = balance(all_samples, min_n, seed=seed)
    bal_path = str(output_dir / "stimuli_with_answers_balanced.jsonl")
    counts_bal = save_jsonl(balanced, bal_path)
    print(f"\n{'='*60}")
    print(f"BALANCED set saved to: {bal_path}")
    print_counts(counts_bal, "Balanced")

    # ---- Medium (50/category) ----
    medium = balance(all_samples, args.medium_n, seed=seed)
    med_path = str(output_dir / "stimuli_with_answers_medium.jsonl")
    counts_med = save_jsonl(medium, med_path)
    print(f"\n{'='*60}")
    print(f"MEDIUM set saved to: {med_path}")
    print_counts(counts_med, "Medium")

    # ---- Discovery / Validation splits ----
    for name, data in [("balanced", balanced), ("medium", medium)]:
        disc, val = discovery_validation_split(data, seed=seed)
        disc_path = str(output_dir / f"stimuli_with_answers_{name}_discovery.jsonl")
        val_path  = str(output_dir / f"stimuli_with_answers_{name}_validation.jsonl")
        counts_d = save_jsonl(disc, disc_path)
        counts_v = save_jsonl(val, val_path)
        print(f"\n{'='*60}")
        print(f"{name.upper()} DISCOVERY split: {disc_path}")
        print_counts(counts_d, f"{name} discovery")
        print(f"{name.upper()} VALIDATION split: {val_path}")
        print_counts(counts_v, f"{name} validation")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Full:                {sum(counts_full.values()):>5d} samples (unbalanced)")
    print(f"  Balanced:            {sum(counts_bal.values()):>5d} samples ({min_n}/cat)")
    print(f"  Medium:              {sum(counts_med.values()):>5d} samples ({args.medium_n}/cat)")
    print(f"  Discovery/Validation splits created for balanced and medium sets")
    print(f"\nAll files in: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
