#!/usr/bin/env python3
"""
Emotion vs Narrative frame: what is the model's REAL internal category?

Design: 4 emotions × 3 narrative frames = 12 scenarios (implicit, no emotion words)
Measure hidden-state similarity → build 12×12 RDM
If organized by EMOTION: anger-moral ≈ anger-threat ≈ anger-interpersonal
If organized by NARRATIVE: anger-moral ≈ disgust-moral ≈ sadness-moral

Also ask the model to classify the emotion and check consistency.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform, cosine

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

EMOTIONS = ["anger", "fear", "sadness", "disgust"]
FRAMES = ["moral", "threat", "interpersonal"]

# 4 emotions × 3 narrative frames, IMPLICIT descriptions (no emotion words)
SCENARIOS = {
    ("anger", "moral"): [
        "After discovering the charity was a scam, he couldn't sit still. He kept pacing, his jaw tight, planning how to expose them.",
        "When the news showed politicians covering up pollution deaths, she slammed her laptop shut and started drafting letters to every newspaper.",
    ],
    ("anger", "threat"): [
        "The car swerved right at his daughter. He jumped forward, fists clenched, ready to drag the driver out.",
        "When the intruder rattled the doorknob again, she grabbed the heaviest thing she could find, her whole body rigid and ready.",
    ],
    ("anger", "interpersonal"): [
        "He found the texts on her phone. His hands were shaking as he put it down, replaying every lie she'd told him.",
        "After her sister took credit for the recipe at Thanksgiving again, she gripped her fork so hard it bent.",
    ],
    ("fear", "moral"): [
        "He sat frozen reading the leaked documents. If this got out, everything he believed about the institution would collapse. His mind raced through the implications.",
        "She kept rereading the whistleblower report, her stomach dropping with each page. What if they came after the people who spoke up?",
    ],
    ("fear", "threat"): [
        "The footsteps behind her got faster when she sped up. She clutched her keys between her fingers and scanned for an open store.",
        "He heard the tornado siren while home alone. His hands fumbled with the basement door, heart hammering so loud he could barely think.",
    ],
    ("fear", "interpersonal"): [
        "Every time he brought up the future, she changed the subject. Lying in bed, he stared at the ceiling, chest tight, wondering if she was already gone in her mind.",
        "His mother's voice on the phone was too calm. She never called on weekdays. He sat down slowly, bracing for whatever came next.",
    ],
    ("sadness", "moral"): [
        "She watched the footage of the demolished refugee camp and put down her coffee. For a long time she just sat there, staring at nothing, feeling something heavy settle in her chest.",
        "He read about the teacher who lost her pension after 30 years due to a policy loophole. He closed the article and sat quietly for a long time.",
    ],
    ("sadness", "threat"): [
        "After the earthquake, he walked through what used to be his neighborhood. Every collapsed wall was a memory. He picked up a child's shoe from the rubble and couldn't move.",
        "The doctor said the treatment wasn't working. She nodded, walked to her car, and sat there for twenty minutes before she could turn the key.",
    ],
    ("sadness", "interpersonal"): [
        "He packed the last box from their apartment. The echo in the empty room hit him harder than the argument ever did. He sat on the floor where the couch used to be.",
        "She found her father's old watch in a drawer. He'd been gone three years. She held it to her ear, just listening.",
    ],
    ("disgust", "moral"): [
        "Reading about the CEO's golden parachute while workers lost their insurance made his skin crawl. He pushed the newspaper away like it was contaminated.",
        "She watched the footage of the senator laughing about the policy that destroyed thousands of families. She had to close the browser; her stomach was turning.",
    ],
    ("disgust", "threat"): [
        "Opening the container in the back of the fridge, the smell hit him and he gagged. Mold had consumed everything, black and green and spreading to the shelf.",
        "She pulled back the hotel sheets and saw the stains and something moving. She backed away, her skin crawling, and wouldn't touch anything else in the room.",
    ],
    ("disgust", "interpersonal"): [
        "He found out his friend had been secretly recording their private conversations to share with others. Every memory of their friendship now felt tainted and wrong.",
        "She discovered her mentor had been manipulating students for years, playing them against each other. Looking at the old photos of them together made her feel physically ill.",
    ],
}


def get_embedding(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    last_hidden = out.hidden_states[-1][0].float()
    mask = inputs["attention_mask"][0].bool()
    emb = last_hidden[mask].mean(dim=0)
    emb = emb / (emb.norm() + 1e-8)
    return emb.cpu().numpy()


def classify_emotion(model, tokenizer, text, device):
    prompt = f'Read the following passage and identify the PRIMARY emotion the person is feeling. Answer with exactly one word (anger, fear, sadness, disgust, happiness, surprise, or other).\n\nPassage: "{text}"\n\nThe primary emotion is:'
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower().split()[0]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Emotion vs Narrative Frame ({model_short}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    # Get embeddings and classifications for each scenario
    labels = []  # (emotion, frame) for each embedding
    embeddings = []
    classifications = []

    for emo in EMOTIONS:
        for frame in FRAMES:
            texts = SCENARIOS[(emo, frame)]
            for ti, text in enumerate(texts):
                emb = get_embedding(model, tokenizer, text, device)
                cls = classify_emotion(model, tokenizer, text, device)
                embeddings.append(emb)
                labels.append((emo, frame))
                classifications.append(cls)
                print(f"  [{emo:>8s} | {frame:>13s}] classified as: {cls:>10s} {'✓' if cls.startswith(emo[:3]) else '✗'}")

    embeddings = np.stack(embeddings)
    n = len(embeddings)

    # Build RDM
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = cosine(embeddings[i], embeddings[j])

    # Key test: within-emotion distance vs within-frame distance
    print(f"\n{'='*60}")
    print("KEY TEST: Is the model organized by emotion or by frame?")
    print(f"{'='*60}")

    same_emo_dists = []
    same_frame_dists = []
    diff_both_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            emo_i, frame_i = labels[i]
            emo_j, frame_j = labels[j]
            d = rdm[i, j]

            if emo_i == emo_j and frame_i != frame_j:
                same_emo_dists.append(d)
            elif frame_i == frame_j and emo_i != emo_j:
                same_frame_dists.append(d)
            elif emo_i != emo_j and frame_i != frame_j:
                diff_both_dists.append(d)

    mean_same_emo = np.mean(same_emo_dists)
    mean_same_frame = np.mean(same_frame_dists)
    mean_diff = np.mean(diff_both_dists)

    print(f"\n  Same EMOTION, different frame:  mean dist = {mean_same_emo:.5f}  (n={len(same_emo_dists)})")
    print(f"  Same FRAME, different emotion:  mean dist = {mean_same_frame:.5f}  (n={len(same_frame_dists)})")
    print(f"  Different both:                 mean dist = {mean_diff:.5f}  (n={len(diff_both_dists)})")

    print(f"\n  INTERPRETATION:")
    if mean_same_emo < mean_same_frame:
        print(f"  Same-emotion pairs are CLOSER → model organized by EMOTION")
        print(f"  (Emotion binds representations more than narrative frame)")
    else:
        print(f"  Same-frame pairs are CLOSER → model organized by NARRATIVE FRAME")
        print(f"  (Narrative context binds representations more than emotion type)")

    ratio = mean_same_frame / mean_same_emo if mean_same_emo > 0 else 0
    print(f"  Ratio (same_frame / same_emotion): {ratio:.3f}")

    # Permutation test
    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(same_emo_dists, same_frame_dists, alternative='two-sided')
    print(f"  Mann-Whitney U: U={U:.0f}, p={p:.6f}")

    # Classification accuracy
    print(f"\n{'='*60}")
    print("EMOTION CLASSIFICATION ACCURACY")
    print(f"{'='*60}")

    correct = sum(1 for (emo, frame), cls in zip(labels, classifications) if cls.startswith(emo[:3]))
    total = len(classifications)
    print(f"\n  Overall accuracy: {correct}/{total} ({correct/total:.0%})")

    # Per emotion accuracy
    for emo in EMOTIONS:
        emo_items = [(l, c) for l, c in zip(labels, classifications) if l[0] == emo]
        emo_correct = sum(1 for l, c in emo_items if c.startswith(emo[:3]))
        print(f"  {emo:>10s}: {emo_correct}/{len(emo_items)} ({emo_correct/len(emo_items):.0%})")

    # Per frame accuracy
    print()
    for frame in FRAMES:
        frame_items = [(l, c) for l, c in zip(labels, classifications) if l[1] == frame]
        frame_correct = sum(1 for (emo, f), c in frame_items if c.startswith(emo[:3]))
        print(f"  {frame:>13s}: {frame_correct}/{len(frame_items)} ({frame_correct/len(frame_items):.0%})")

    # Classification confusion: what does the model label things as?
    print(f"\n  Classification matrix:")
    print(f"  {'True':>10s} | Classified as...")
    for emo in EMOTIONS:
        cls_list = [c for (e, f), c in zip(labels, classifications) if e == emo]
        from collections import Counter
        counts = Counter(cls_list)
        print(f"  {emo:>10s} | {dict(counts)}")

    # Visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MDS of the 24 scenarios
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(rdm)

    emo_colors = {"anger": "#e74c3c", "fear": "#8e44ad", "sadness": "#2980b9", "disgust": "#27ae60"}
    frame_markers = {"moral": "o", "threat": "^", "interpersonal": "s"}

    ax = axes[0]
    for i, (emo, frame) in enumerate(labels):
        ax.scatter(coords[i, 0], coords[i, 1], c=emo_colors[emo],
                  marker=frame_markers[frame], s=100, edgecolors="white", linewidth=0.5)
    # Legend
    for emo, color in emo_colors.items():
        ax.scatter([], [], c=color, label=emo, s=60)
    for frame, marker in frame_markers.items():
        ax.scatter([], [], c="gray", marker=marker, label=frame, s=60)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("MDS: colored by emotion, shaped by frame")
    ax.grid(alpha=0.2)

    # Bar chart: same-emotion vs same-frame distances
    ax = axes[1]
    bars = ax.bar(["Same emotion\ndiff frame", "Same frame\ndiff emotion", "Different\nboth"],
                  [mean_same_emo, mean_same_frame, mean_diff],
                  color=["#2e5cb8", "#e8743b", "#999"])
    for b, v in zip(bars, [mean_same_emo, mean_same_frame, mean_diff]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean cosine distance")
    ax.set_title(f"Organized by {'EMOTION' if mean_same_emo < mean_same_frame else 'FRAME'}? (p={p:.4f})")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "emotion_vs_narrative.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {FIG_DIR / 'emotion_vs_narrative.png'}")

    # Save results
    results = {
        "model": model_short,
        "mean_same_emotion_dist": float(mean_same_emo),
        "mean_same_frame_dist": float(mean_same_frame),
        "mean_diff_both_dist": float(mean_diff),
        "organized_by": "emotion" if mean_same_emo < mean_same_frame else "frame",
        "mann_whitney_p": float(p),
        "classification_accuracy": correct / total,
        "per_scenario": [
            {"emotion": emo, "frame": frame, "classified_as": cls}
            for (emo, frame), cls in zip(labels, classifications)
        ],
    }
    json.dump(results, open(OUT_DIR / f"emotion_vs_narrative_{model_short}.json", "w"), indent=2)
    print(f"Saved results")
    print("DONE")


if __name__ == "__main__":
    main()
