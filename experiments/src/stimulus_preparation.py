"""
Phase 0: Prepare stimulus dataset (10,000 samples across 25 categories).

Sources: HuggingFace datasets (GSM8K, MATH, MMLU, HumanEval, etc.) + custom templates.
Output: JSONL file with fields {text, category, subcategory, source}.
"""

import json
import os
import random
from pathlib import Path
from datasets import load_dataset


CUSTOM_TEMPLATES = {
    "grammar_correction": [
        "Please correct the grammar in the following sentence: '{sent}'",
        "Fix the grammatical errors: '{sent}'",
    ],
    "creative_writing": [
        "Write a short poem about {topic}.",
        "Write a creative short story (100 words) about {topic}.",
        "Compose a haiku about {topic}.",
    ],
    "formal_writing": [
        "Write a formal email to {recipient} about {topic}.",
        "Draft a professional report summary on {topic}.",
    ],
    "dialogue": [
        "Continue the following conversation:\nUser: {prompt}\nAssistant:",
        "Write a dialogue between a teacher and student about {topic}.",
    ],
    "causal_reasoning": [
        "What would happen if {event}? Explain the causal chain.",
        "Why does {phenomenon} occur? Explain step by step.",
    ],
    "code_debugging": [
        "Find and fix the bug in this code:\n```python\n{code}\n```",
    ],
    "code_explanation": [
        "Explain what this code does:\n```python\n{code}\n```",
    ],
    "algorithm_design": [
        "Design an efficient algorithm to {task}. Provide pseudocode.",
    ],
    "current_events": [
        "Analyze the implications of {event} from multiple perspectives.",
    ],
    "domain_expertise": [
        "As a {role}, explain {topic} to a non-specialist.",
    ],
    "ethical_reasoning": [
        "Is it ethical to {action}? Discuss from multiple moral frameworks.",
        "Analyze the following ethical dilemma: {dilemma}",
    ],
    "analogical_reasoning": [
        "{a} is to {b} as {c} is to what? Explain your reasoning.",
        "Draw an analogy between {domain1} and {domain2}.",
    ],
    "planning": [
        "Create a step-by-step plan to {goal}.",
        "Break down the task of {task} into subtasks and dependencies.",
    ],
    "metacognition": [
        "How confident are you in answering questions about {topic}? What would you need to know?",
        "What are the limits of your knowledge about {topic}?",
    ],
    "role_playing": [
        "You are a {character}. Respond to: {prompt}",
        "Pretend you are a {role} in {setting}. What would you do if {situation}?",
    ],
    "humor_sarcasm": [
        "Explain why this joke is funny: '{joke}'",
        "Write a witty response to: '{prompt}'",
        "Is the following statement sarcastic? '{statement}' Explain.",
    ],
}

FILL_DATA = {
    "topics": ["quantum physics", "climate change", "ancient Rome", "machine learning",
               "the ocean", "democracy", "music theory", "space exploration",
               "cooking", "friendship", "time travel", "artificial intelligence"],
    "recipients": ["a professor", "a hiring manager", "a client", "a colleague"],
    "events": ["gravity suddenly doubled", "the internet disappeared for a month",
               "humans could photosynthesize", "all languages merged into one"],
    "phenomena": ["the sky is blue", "water expands when frozen", "economies cycle between growth and recession"],
    "roles": ["doctor", "lawyer", "engineer", "biologist", "economist", "historian"],
    "characters": ["Sherlock Holmes", "a medieval knight", "a pirate captain", "Albert Einstein"],
    "dilemmas": ["a self-driving car must choose between two harmful outcomes",
                 "using AI to predict criminal behavior before crimes are committed"],
    "jokes": ["Why do programmers prefer dark mode? Because light attracts bugs.",
              "A SQL query walks into a bar, sees two tables and asks: can I join you?"],
    "grammar_errors": [
        "Me and him went to the store yesterday and buyed some foods.",
        "The team have decided that they is going to the competition.",
        "She don't know nothing about the situation what happened.",
    ],
    "buggy_code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-3)",
        "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1",
    ],
    "algorithm_tasks": [
        "find the longest common subsequence of two strings",
        "detect cycles in a directed graph",
        "merge K sorted linked lists efficiently",
    ],
}


def generate_custom_samples(category, n=400, seed=42):
    """Generate samples from templates for custom categories."""
    rng = random.Random(seed)
    templates = CUSTOM_TEMPLATES.get(category, [])
    if not templates:
        return []

    samples = []
    for i in range(n):
        tpl = rng.choice(templates)
        text = tpl
        for key in ["topic", "sent", "prompt", "event", "phenomenon", "recipient",
                     "role", "character", "dilemma", "joke", "statement",
                     "code", "task", "action", "goal", "setting", "situation",
                     "a", "b", "c", "domain1", "domain2"]:
            if "{" + key + "}" in text:
                if key == "sent":
                    text = text.replace("{sent}", rng.choice(FILL_DATA["grammar_errors"]))
                elif key == "code":
                    text = text.replace("{code}", rng.choice(FILL_DATA["buggy_code"]))
                elif key == "task":
                    text = text.replace("{task}", rng.choice(FILL_DATA["algorithm_tasks"]))
                elif key in ["topic", "goal", "prompt", "action", "situation"]:
                    text = text.replace("{" + key + "}", rng.choice(FILL_DATA["topics"]))
                elif key in ["event"]:
                    text = text.replace("{event}", rng.choice(FILL_DATA["events"]))
                elif key in ["phenomenon"]:
                    text = text.replace("{phenomenon}", rng.choice(FILL_DATA["phenomena"]))
                elif key in ["recipient"]:
                    text = text.replace("{recipient}", rng.choice(FILL_DATA["recipients"]))
                elif key in ["role"]:
                    text = text.replace("{role}", rng.choice(FILL_DATA["roles"]))
                elif key in ["character"]:
                    text = text.replace("{character}", rng.choice(FILL_DATA["characters"]))
                elif key in ["dilemma"]:
                    text = text.replace("{dilemma}", rng.choice(FILL_DATA["dilemmas"]))
                elif key in ["joke", "statement"]:
                    text = text.replace("{" + key + "}", rng.choice(FILL_DATA["jokes"]))
                elif key in ["setting"]:
                    text = text.replace("{setting}", rng.choice(["a hospital", "a courtroom", "a spaceship"]))
                elif key in ["domain1", "domain2", "a", "b", "c"]:
                    text = text.replace("{" + key + "}", rng.choice(FILL_DATA["topics"]))

        samples.append({"text": text, "category": category, "source": "custom"})

    return samples


def load_hf_samples(dataset_name, category, n=400, seed=42):
    """Load samples from HuggingFace datasets."""
    rng = random.Random(seed)
    samples = []

    try:
        if dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="train")
            items = rng.sample(list(ds), min(n, len(ds)))
            for item in items:
                samples.append({"text": item["question"], "category": category, "source": "gsm8k"})

        elif dataset_name == "math":
            ds = load_dataset("hendrycks/competition_math", split="train")
            items = rng.sample(list(ds), min(n, len(ds)))
            for item in items:
                samples.append({"text": item["problem"], "category": category, "source": "math"})

        elif dataset_name == "logiqa":
            ds = load_dataset("lucasmccabe/logiqa", split="train")
            items = rng.sample(list(ds), min(n, len(ds)))
            for item in items:
                text = item["context"] + "\nQuestion: " + item["query"]
                samples.append({"text": text, "category": category, "source": "logiqa"})

        elif dataset_name == "arc_challenge":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
            items = rng.sample(list(ds), min(n, len(ds)))
            for item in items:
                samples.append({"text": item["question"], "category": category, "source": "arc"})

        elif dataset_name == "humaneval+mbpp":
            ds1 = load_dataset("openai/openai_humaneval", split="test")
            items = list(ds1)
            for item in items[:min(n // 2, len(items))]:
                samples.append({"text": item["prompt"], "category": category, "source": "humaneval"})
            remaining = n - len(samples)
            if remaining > 0:
                ds2 = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
                items2 = rng.sample(list(ds2), min(remaining, len(ds2)))
                for item in items2:
                    samples.append({"text": item["text"], "category": category, "source": "mbpp"})

        elif dataset_name.startswith("mmlu_"):
            subject_group = dataset_name.replace("mmlu_", "")
            ds = load_dataset("cais/mmlu", "all", split="test")
            subject_map = {
                "stem": ["physics", "chemistry", "biology", "computer_science", "mathematics",
                         "electrical_engineering", "astronomy", "machine_learning"],
                "humanities": ["philosophy", "history", "law", "moral_scenarios",
                               "world_religions", "logical_fallacies"],
                "professional": ["professional_medicine", "professional_law",
                                 "professional_accounting", "clinical_knowledge"],
            }
            subjects = subject_map.get(subject_group, [])
            filtered = [x for x in ds if x.get("subject", "") in subjects]
            if not filtered:
                filtered = list(ds)
            items = rng.sample(filtered, min(n, len(filtered)))
            for item in items:
                text = item["question"] + "\nChoices: " + ", ".join(
                    [f"({chr(65+i)}) {c}" for i, c in enumerate(item["choices"])]
                )
                samples.append({"text": text, "category": category, "source": f"mmlu_{subject_group}"})

        elif dataset_name == "xsum":
            ds = load_dataset("EdinburghNLP/xsum", split="train")
            items = rng.sample(list(ds), min(n, len(ds)))
            for item in items:
                text = f"Summarize the following article:\n\n{item['document'][:1000]}"
                samples.append({"text": text, "category": category, "source": "xsum"})

        elif dataset_name == "flores":
            ds = load_dataset("facebook/flores", "eng_Latn-zho_Hans", split="devtest")
            items = rng.sample(list(ds), min(n, len(ds)))
            for item in items:
                text = f"Translate the following English text to Chinese:\n\n{item['sentence_eng_Latn']}"
                samples.append({"text": text, "category": category, "source": "flores"})

        elif dataset_name == "ethics":
            ds = load_dataset("hendrycks/ethics", "commonsense", split="train")
            items = rng.sample(list(ds), min(n, len(ds)))
            for item in items:
                text = f"Is the following action ethical? Explain your reasoning.\n\n{item['input']}"
                samples.append({"text": text, "category": category, "source": "ethics"})

    except Exception as e:
        print(f"Warning: Failed to load {dataset_name}: {e}. Using custom fallback.")
        return generate_custom_samples(category, n, seed)

    while len(samples) < n:
        extras = generate_custom_samples(category, n - len(samples), seed + len(samples))
        samples.extend(extras)

    return samples[:n]


CATEGORY_SOURCES = {
    "grammar_correction": "custom",
    "translation": "flores",
    "summarization": "xsum",
    "creative_writing": "custom",
    "formal_writing": "custom",
    "dialogue": "custom",
    "math_word_problems": "gsm8k",
    "formal_math": "math",
    "logical_reasoning": "logiqa",
    "commonsense": "arc_challenge",
    "causal_reasoning": "custom",
    "code_generation": "humaneval+mbpp",
    "code_debugging": "custom",
    "code_explanation": "custom",
    "algorithm_design": "custom",
    "science_qa": "mmlu_stem",
    "history_geography": "mmlu_humanities",
    "current_events": "custom",
    "domain_expertise": "mmlu_professional",
    "ethical_reasoning": "ethics",
    "analogical_reasoning": "custom",
    "planning": "custom",
    "metacognition": "custom",
    "role_playing": "custom",
    "humor_sarcasm": "custom",
}


def prepare_all_stimuli(output_path, n_per_category=400, seed=42):
    """Prepare the full stimulus dataset."""
    all_samples = []

    for category, source in CATEGORY_SOURCES.items():
        print(f"Preparing {category} (source: {source})...")
        if source == "custom":
            samples = generate_custom_samples(category, n_per_category, seed)
        else:
            samples = load_hf_samples(source, category, n_per_category, seed)

        for s in samples:
            s["category"] = category

        all_samples.extend(samples)
        print(f"  → {len(samples)} samples")

    random.Random(seed).shuffle(all_samples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_samples)} samples across {len(CATEGORY_SOURCES)} categories")
    print(f"Saved to: {output_path}")

    # Also save category index
    index = {}
    for i, s in enumerate(all_samples):
        cat = s["category"]
        if cat not in index:
            index[cat] = []
        index[cat].append(i)

    index_path = output_path.replace(".jsonl", "_index.json")
    with open(index_path, "w") as f:
        json.dump({k: len(v) for k, v in index.items()}, f, indent=2)

    return all_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/data/user/mzhang630/data/nature_exp/stimuli/all_stimuli.jsonl")
    parser.add_argument("--n_per_category", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_all_stimuli(args.output, args.n_per_category, args.seed)
