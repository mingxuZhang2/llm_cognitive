#!/usr/bin/env python3
"""
C: Mechanistic analysis — at which layer does narrative frame vs emotion
category become the dominant organizing principle?

For each layer:
  1. Decode narrative frame (moral/threat/interpersonal) from hidden states
  2. Decode emotion category (anger/fear/sadness/disgust) from hidden states
  3. Compare accuracy: which is easier to decode at each layer?

This reveals the PROCESSING ORDER: does the model identify narrative frame
first and emotion second, or vice versa?

GPU needed (forward passes through all layers).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

EMOTIONS = ["anger", "fear", "sadness", "disgust"]
FRAMES = ["moral", "threat", "interpersonal"]

# Same scenarios as emotion_vs_narrative.py
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
        "He sat frozen reading the leaked documents. If this got out, everything he believed about the institution would collapse.",
        "She kept rereading the whistleblower report, her stomach dropping with each page. What if they came after the people who spoke up?",
    ],
    ("fear", "threat"): [
        "The footsteps behind her got faster when she sped up. She clutched her keys between her fingers and scanned for an open store.",
        "He heard the tornado siren while home alone. His hands fumbled with the basement door, heart hammering.",
    ],
    ("fear", "interpersonal"): [
        "Every time he brought up the future, she changed the subject. Lying in bed, he stared at the ceiling, chest tight.",
        "His mother's voice on the phone was too calm. She never called on weekdays. He sat down slowly, bracing.",
    ],
    ("sadness", "moral"): [
        "She watched the footage of the demolished refugee camp and put down her coffee. She just sat there, staring at nothing.",
        "He read about the teacher who lost her pension after 30 years due to a policy loophole. He closed the article and sat quietly.",
    ],
    ("sadness", "threat"): [
        "After the earthquake, he walked through what used to be his neighborhood. He picked up a child's shoe from the rubble.",
        "The doctor said the treatment wasn't working. She nodded, walked to her car, and sat there for twenty minutes.",
    ],
    ("sadness", "interpersonal"): [
        "He packed the last box from their apartment. The echo in the empty room hit him harder than any argument.",
        "She found her father's old watch in a drawer. He'd been gone three years. She held it to her ear, just listening.",
    ],
    ("disgust", "moral"): [
        "Reading about the CEO's golden parachute while workers lost insurance made his skin crawl. He pushed the paper away.",
        "She watched the senator laughing about the policy that destroyed families. She had to close the browser, stomach turning.",
    ],
    ("disgust", "threat"): [
        "Opening the container in the back of the fridge, the smell hit him and he gagged. Mold everywhere, black and green.",
        "She pulled back the hotel sheets and saw stains and something moving. She backed away, skin crawling.",
    ],
    ("disgust", "interpersonal"): [
        "He found out his friend had been secretly recording their private conversations to share. Every memory felt tainted.",
        "She discovered her mentor had been manipulating students for years. Looking at old photos together made her feel ill.",
    ],
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Emotion Mechanistic Analysis ({model_short}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    # Get hidden states at ALL layers for all stimuli
    print("\n--- Extracting hidden states at all layers ---")
    all_hidden = []  # list of (n_layers, hidden_dim) per stimulus
    all_emotions = []
    all_frames = []

    for emo in EMOTIONS:
        for frame in FRAMES:
            for text in SCENARIOS[(emo, frame)]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                # Mean pool each layer
                mask = inputs["attention_mask"][0].bool()
                layer_acts = []
                for layer_hidden in out.hidden_states[1:]:  # skip embedding
                    act = layer_hidden[0][mask].float().mean(dim=0).cpu().numpy()
                    layer_acts.append(act)
                all_hidden.append(np.stack(layer_acts))  # (n_layers, hidden_dim)
                all_emotions.append(emo)
                all_frames.append(frame)

    all_hidden = np.stack(all_hidden)  # (24, n_layers, hidden_dim)
    n_stim, n_layers, hidden_dim = all_hidden.shape
    print(f"  Shape: {all_hidden.shape} ({n_stim} stimuli, {n_layers} layers, {hidden_dim} dims)")

    emotion_labels = np.array([EMOTIONS.index(e) for e in all_emotions])
    frame_labels = np.array([FRAMES.index(f) for f in all_frames])

    # Layer-wise decoding
    print("\n--- Layer-wise decoding: emotion vs frame ---")
    print(f"  {'Layer':>6s} | {'Emotion acc':>11s} | {'Frame acc':>10s} | {'Dominant':>10s}")
    print(f"  {'-'*50}")

    emotion_accs = []
    frame_accs = []

    for li in range(n_layers):
        X = all_hidden[:, li, :]
        X_scaled = StandardScaler().fit_transform(X)

        # Emotion classification (4-way)
        clf_emo = LogisticRegression(max_iter=1000, C=1.0)
        emo_scores = cross_val_score(clf_emo, X_scaled, emotion_labels, cv=min(5, n_stim//4), scoring="accuracy")
        emo_acc = emo_scores.mean()

        # Frame classification (3-way)
        clf_frame = LogisticRegression(max_iter=1000, C=1.0)
        frame_scores = cross_val_score(clf_frame, X_scaled, frame_labels, cv=min(5, n_stim//4), scoring="accuracy")
        frame_acc = frame_scores.mean()

        emotion_accs.append(emo_acc)
        frame_accs.append(frame_acc)

        dominant = "FRAME" if frame_acc > emo_acc else "EMOTION"
        if li % 4 == 0 or li == n_layers - 1:
            print(f"  L{li:>4d} | {emo_acc:11.1%} | {frame_acc:10.1%} | {dominant:>10s}")

    # Summary
    print(f"\n--- Summary ---")
    emo_peak = max(emotion_accs)
    frame_peak = max(frame_accs)
    emo_peak_layer = emotion_accs.index(emo_peak)
    frame_peak_layer = frame_accs.index(frame_peak)
    print(f"  Emotion peak: {emo_peak:.1%} at L{emo_peak_layer}")
    print(f"  Frame peak:   {frame_peak:.1%} at L{frame_peak_layer}")

    # Where does frame first exceed emotion?
    frame_dominant_layers = [li for li in range(n_layers) if frame_accs[li] > emotion_accs[li]]
    if frame_dominant_layers:
        print(f"  Frame first dominates at: L{frame_dominant_layers[0]}")
        print(f"  Frame dominant in {len(frame_dominant_layers)}/{n_layers} layers")

    # Chance levels
    emo_chance = 1.0 / len(EMOTIONS)
    frame_chance = 1.0 / len(FRAMES)
    print(f"  Chance: emotion={emo_chance:.1%}, frame={frame_chance:.1%}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = range(n_layers)
    ax.plot(layers, emotion_accs, "-o", color="#e74c3c", lw=2, ms=4, label=f"Emotion category (4-way, chance={emo_chance:.0%})")
    ax.plot(layers, frame_accs, "-s", color="#2e5cb8", lw=2, ms=4, label=f"Narrative frame (3-way, chance={frame_chance:.0%})")
    ax.axhline(emo_chance, color="#e74c3c", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(frame_chance, color="#2e5cb8", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title(f"Layer-wise decoding: Emotion vs Narrative Frame ({model_short})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "emotion_mechanistic_layers.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {FIG_DIR / 'emotion_mechanistic_layers.png'}")

    results = {
        "model": model_short,
        "n_stimuli": n_stim,
        "n_layers": n_layers,
        "emotion_accs": [float(a) for a in emotion_accs],
        "frame_accs": [float(a) for a in frame_accs],
        "emotion_peak": {"acc": float(emo_peak), "layer": emo_peak_layer},
        "frame_peak": {"acc": float(frame_peak), "layer": frame_peak_layer},
    }
    json.dump(results, open(OUT_DIR / f"emotion_mechanistic_{model_short}.json", "w"), indent=2)
    print(f"Saved results")
    print("DONE")


if __name__ == "__main__":
    main()
