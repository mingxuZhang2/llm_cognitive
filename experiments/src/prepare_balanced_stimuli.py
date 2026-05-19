"""
Prepare balanced stimulus sets from the full (unbalanced) stimuli_full.jsonl.

The full set has 3133 samples across 8 categories with unequal counts:
  code=164, ethics=817, factual_qa=400, humanities=400,
  language=400, math=400, reasoning=400, science=152

This script creates:
  1. stimuli_balanced.jsonl  — subsample each category to match the smallest
                               category (science=152), producing ~1216 samples
  2. stimuli_medium.jsonl    — 50 samples per category (400 total) for fast iteration

Both files use the same JSONL format: {text, category, source, idx}
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def load_stimuli(path):
    """Load JSONL stimuli file."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def balance_stimuli(samples, n_per_category, seed=42):
    """Subsample each category to n_per_category, shuffle, re-index."""
    rng = random.Random(seed)

    # Group by category
    by_cat = defaultdict(list)
    for s in samples:
        by_cat[s["category"]].append(s)

    balanced = []
    for cat in sorted(by_cat.keys()):
        pool = by_cat[cat]
        if len(pool) <= n_per_category:
            # Use all samples if category is smaller than target
            balanced.extend(pool)
        else:
            balanced.extend(rng.sample(pool, n_per_category))

    # Shuffle and re-index
    rng.shuffle(balanced)
    for i, s in enumerate(balanced):
        s["idx"] = i

    return balanced


def save_stimuli(samples, output_path):
    """Save samples to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Also save index
    counts = Counter(s["category"] for s in samples)
    index_path = output_path.replace(".jsonl", "_index.json")
    with open(index_path, "w") as f:
        json.dump(dict(sorted(counts.items())), f, indent=2)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Create balanced stimulus sets from unbalanced full stimuli."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to stimuli_full.jsonl"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write balanced output files"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--medium_n", type=int, default=50,
        help="Samples per category for the medium set (default: 50)"
    )
    args = parser.parse_args()

    # Load
    print(f"Loading stimuli from {args.input}...")
    samples = load_stimuli(args.input)
    print(f"  Loaded {len(samples)} samples")

    # Category distribution
    counts = Counter(s["category"] for s in samples)
    print(f"\nOriginal distribution ({len(counts)} categories):")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:35s} {n:>4d}")
    min_cat = min(counts, key=counts.get)
    min_n = counts[min_cat]
    print(f"\n  Smallest category: {min_cat} = {min_n}")

    # === Balanced set: match smallest category ===
    balanced_path = str(Path(args.output_dir) / "stimuli_balanced.jsonl")
    print(f"\n{'='*60}")
    print(f"Creating BALANCED set ({min_n}/category)...")
    balanced = balance_stimuli(samples, n_per_category=min_n, seed=args.seed)
    bal_counts = save_stimuli(balanced, balanced_path)
    print(f"  Total: {len(balanced)} samples")
    for cat, n in sorted(bal_counts.items()):
        print(f"    {cat:35s} {n:>4d}")
    print(f"  Saved to: {balanced_path}")

    # === Medium set: 50/category ===
    medium_path = str(Path(args.output_dir) / "stimuli_medium.jsonl")
    print(f"\n{'='*60}")
    print(f"Creating MEDIUM set ({args.medium_n}/category)...")
    medium = balance_stimuli(samples, n_per_category=args.medium_n, seed=args.seed)
    med_counts = save_stimuli(medium, medium_path)
    print(f"  Total: {len(medium)} samples")
    for cat, n in sorted(med_counts.items()):
        print(f"    {cat:35s} {n:>4d}")
    print(f"  Saved to: {medium_path}")

    print(f"\n{'='*60}")
    print("Done. Summary:")
    print(f"  Full (original):  {len(samples):>5d} samples (unbalanced)")
    print(f"  Balanced:         {len(balanced):>5d} samples ({min_n}/category)")
    print(f"  Medium:           {len(medium):>5d} samples ({args.medium_n}/category)")


if __name__ == "__main__":
    main()
