"""
Prepare subcategory stimuli for fine-grained dissociation experiments.

Addresses the criticism that 8 broad categories may be distinguished by
surface features (formatting, vocabulary) rather than genuine cognitive
functions. Creates format-matched subcategory splits within categories.

Reads stimuli_full.jsonl and produces stimuli_subcategory.jsonl where
the "category" field is set to the subcategory name (so multi_function_dissociation.py
can consume it directly).

Subcategory splits (12 total):
  - science -> science_factual (recall: "what is X"), science_quantitative (calculation/measurement)
  - humanities -> history, philosophy
  - math -> arithmetic, word_problem
  - language -> procedural, activity_narration
  - code -> code (unchanged)
  - reasoning -> reasoning (unchanged)
  - factual_qa -> factual_qa (unchanged)
  - ethics -> ethics (unchanged)

Key design: subcategory pairs share format (both multiple-choice, both
word problems, etc.) but differ in cognitive demand. If the model has
separate neural modules for factual recall vs quantitative reasoning
WITHIN the same stimulus format, that rules out surface-feature confounds.

Minimum 30 samples per subcategory, balanced where possible.
"""

import json
import re
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# Keyword lists for subcategory classification
# ============================================================

# Science split: factual recall vs causal explanation
# Both are MMLU STEM multiple-choice (same format), but different cognitive demands:
# - science_factual: "What is X?", "Which is Y?" -> factual recall
# - science_explanation: "Why does X?", "How does Y work?" -> causal reasoning
# This is the strongest format-controlled split: identical stimulus format (MC),
# but one requires recall and the other requires mechanistic understanding.

PHILOSOPHY_KEYWORDS = [
    "moral", "ethic", "utilitari", "virtue", "kant", "argument",
    "justice", "morally", "scenario", "duty", "consequential",
    "deontolog", "categorical", "hume", "locke", "sartre",
    "parfit", "singer", "berkeley", "stevenson", "mill",
    "existentialism", "empiric", "passions", "obligation",
    "famine", "rights", "wrong", "right thing", "illusion",
    "sensation", "ought", "normative", "relativism", "subjective",
    "objective", "philosophy", "philosopher", "utilitarian",
    "hedonism", "rational", "reason alone", "act of",
    "clearly morally wrong",  # common MMLU moral_scenarios prefix
]

HISTORY_KEYWORDS = [
    "history", "civilization", "empire", "war", "ancient", "century",
    "dynasty", "kingdom", "colony", "revolution", "archaeolog",
    "prehistoric", "medieval", "independence", "roman", "greek",
    "egyptian", "conquest", "reformation", "stone tools", "artifact",
    "bone", "excavat", "village", "settlement", "homeland",
    "migration", "dharma", "dalai", "religion", "buddhis",
    "christia", "islam", "tradition", "ritual", "tribe",
    "indigenous", "culture change", "cultural", "subsistence",
    "foraging", "agriculture", "crop", "domesticat", "pottery",
    "bronze", "iron age", "paleolith", "neolith", "mesolithic",
    "hoabinhian", "catalhoyuk", "mound", "chiefdom",
]

# Hellaswag prefixes that indicate procedural how-to content
PROCEDURAL_PREFIXES = {
    "Computers and Electronics", "Home and Garden", "Health",
    "Food and Entertaining", "Pets and Animals", "Youth",
    "Finance and Business", "Education and Communications",
    "Work World", "Travel", "Cars & Other Vehicles",
    "Relationships", "Holidays and Traditions",
}


# ============================================================
# Classification functions
# ============================================================

def has_keyword(text, keywords):
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def classify_science(sample):
    """Classify a science sample as science_explanation or science_factual.

    Both are MMLU STEM multiple-choice (same format), but:
    - science_explanation: "Why does X?", "How does Y work?" -> causal reasoning
    - science_factual: "What is X?", "Which is Y?" -> factual recall
    """
    text = sample["text"].strip().lower()
    is_explanation = (
        text.startswith("why ") or
        text.startswith("how do") or
        text.startswith("how does") or
        text.startswith("how did") or
        text.startswith("how is") or
        text.startswith("how are") or
        text.startswith("how can") or
        text.startswith("how could") or
        text.startswith("how would") or
        "what is the reason" in text or
        "what causes" in text or
        "what caused" in text or
        "what explains" in text or
        "what would happen" in text or
        "what effect" in text
    )
    return "science_explanation" if is_explanation else "science_factual"


def classify_humanities(sample):
    """Classify a humanities sample as history or philosophy."""
    text = sample["text"]
    has_phil = has_keyword(text, PHILOSOPHY_KEYWORDS)
    has_hist = has_keyword(text, HISTORY_KEYWORDS)

    if has_phil and not has_hist:
        return "philosophy"
    elif has_hist and not has_phil:
        return "history"
    elif has_phil and has_hist:
        phil_count = sum(1 for kw in PHILOSOPHY_KEYWORDS if kw.lower() in text.lower())
        hist_count = sum(1 for kw in HISTORY_KEYWORDS if kw.lower() in text.lower())
        return "philosophy" if phil_count > hist_count else "history"
    else:
        # Check for common patterns of MMLU moral_scenarios format
        if "scenario" in text.lower() or "clearly morally" in text.lower():
            return "philosophy"
        # Default to history for unclassified
        return "history"


def classify_math(sample):
    """Classify a math (GSM8K) sample as arithmetic or word_problem.

    Heuristic: shorter problems with simple setups are arithmetic;
    longer problems with multi-step reasoning are word_problems.
    Also count sentence count as a complexity proxy.
    """
    text = sample["text"]
    n_chars = len(text)
    # Count sentences (rough proxy for reasoning steps)
    n_sentences = text.count(".") + text.count("?")

    # Short text with few sentences -> arithmetic
    # Long text with many sentences -> word_problem
    if n_chars <= 180 and n_sentences <= 3:
        return "arithmetic"
    elif n_chars > 250 or n_sentences > 4:
        return "word_problem"
    else:
        # Middle ground: use char count as tiebreaker
        return "arithmetic" if n_chars <= 200 else "word_problem"


def classify_language(sample):
    """Classify a language (HellaSwag) sample as procedural or activity_narration."""
    text = sample["text"]
    # Extract prefix before first colon
    if ":" in text[:80]:
        prefix = text.split(":")[0].strip()
        if prefix in PROCEDURAL_PREFIXES:
            return "procedural"
    # Check for [header]/[step] markers typical of how-to text
    if "[header]" in text or "[step]" in text:
        return "procedural"
    return "activity_narration"


def classify_sample(sample):
    """Assign a subcategory to a sample based on its original category."""
    cat = sample["category"]

    if cat == "science":
        return classify_science(sample)
    elif cat == "humanities":
        return classify_humanities(sample)
    elif cat == "math":
        return classify_math(sample)
    elif cat == "language":
        return classify_language(sample)
    else:
        # code, reasoning, factual_qa, ethics: keep as-is
        return cat


# ============================================================
# Main pipeline
# ============================================================

def load_stimuli(path):
    """Load JSONL stimuli file."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def prepare_subcategory_stimuli(
    input_path, output_path, min_per_subcat=30, target_per_subcat=50, seed=42
):
    """
    Read stimuli_full.jsonl, classify into subcategories,
    balance to target_per_subcat (or min available), save output.
    """
    rng = random.Random(seed)

    # Load
    print(f"Loading stimuli from {input_path}...")
    samples = load_stimuli(input_path)
    print(f"  Loaded {len(samples)} samples")

    # Show original distribution
    orig_counts = Counter(s["category"] for s in samples)
    print(f"\nOriginal distribution ({len(orig_counts)} categories):")
    for cat, n in sorted(orig_counts.items()):
        print(f"  {cat:30s} {n:>4d}")

    # Classify into subcategories
    print(f"\nClassifying into subcategories...")
    by_subcat = defaultdict(list)
    for s in samples:
        subcat = classify_sample(s)
        s_copy = dict(s)
        s_copy["original_category"] = s["category"]
        s_copy["category"] = subcat  # overwrite category with subcategory
        s_copy["subcategory"] = subcat
        by_subcat[subcat].append(s_copy)

    print(f"\nSubcategory distribution ({len(by_subcat)} subcategories):")
    for subcat, items in sorted(by_subcat.items()):
        orig_cat = items[0]["original_category"]
        print(f"  {subcat:30s} {len(items):>4d}  (from {orig_cat})")

    # Check minimum threshold
    too_small = {k: v for k, v in by_subcat.items() if len(v) < min_per_subcat}
    if too_small:
        print(f"\nWARNING: Subcategories below minimum ({min_per_subcat}):")
        for k, v in sorted(too_small.items()):
            print(f"  {k}: {len(v)} samples")

    # Balance: subsample large categories to target, keep small ones as-is
    balanced = []
    print(f"\nBalancing to {target_per_subcat} samples per subcategory...")
    for subcat in sorted(by_subcat.keys()):
        pool = by_subcat[subcat]
        if len(pool) > target_per_subcat:
            selected = rng.sample(pool, target_per_subcat)
        else:
            selected = list(pool)
        balanced.extend(selected)
        print(f"  {subcat:30s} {len(selected):>4d} / {len(pool):>4d}")

    # Shuffle and re-index
    rng.shuffle(balanced)
    for i, s in enumerate(balanced):
        s["idx"] = i

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in balanced:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save index
    final_counts = Counter(s["category"] for s in balanced)
    index_path = output_path.replace(".jsonl", "_index.json")
    with open(index_path, "w") as f:
        json.dump(dict(sorted(final_counts.items())), f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(balanced)}")
    print(f"Subcategories: {len(final_counts)}")
    print(f"\nFinal distribution:")
    for subcat, n in sorted(final_counts.items()):
        print(f"  {subcat:30s} {n:>4d}")
    print(f"\nSaved to: {output_path}")
    print(f"Index:    {index_path}")

    # Verify all subcategories meet minimum
    all_ok = all(n >= min_per_subcat for n in final_counts.values())
    if all_ok:
        print(f"\nAll subcategories meet minimum threshold ({min_per_subcat}). READY.")
    else:
        small = {k: v for k, v in final_counts.items() if v < min_per_subcat}
        print(f"\nWARNING: {len(small)} subcategories below minimum:")
        for k, v in sorted(small.items()):
            print(f"  {k}: {v}")

    return balanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create subcategory stimuli for fine-grained dissociation."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to stimuli_full.jsonl"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for output stimuli_subcategory.jsonl"
    )
    parser.add_argument(
        "--min_per_subcat", type=int, default=30,
        help="Minimum samples per subcategory (default: 30)"
    )
    parser.add_argument(
        "--target_per_subcat", type=int, default=50,
        help="Target samples per subcategory for balancing (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    prepare_subcategory_stimuli(
        input_path=args.input,
        output_path=args.output,
        min_per_subcat=args.min_per_subcat,
        target_per_subcat=args.target_per_subcat,
        seed=args.seed,
    )
