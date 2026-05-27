#!/usr/bin/env python3
"""
Baseline controls for brain-LLM RSA.

1. GloVe baseline: RSA using static word embeddings (no context, no training)
2. TF-IDF baseline: RSA using bag-of-words features
3. Condition-name baseline: RSA using only the condition label text
4. Sentence-length baseline: RSA using only sentence length as feature

If any baseline achieves ρ close to our trained LLM result (~0.63),
it means the alignment is driven by surface features, not cognitive representations.

Usage:
  python baseline_controls.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path("/hpc2hdd/home/mzhang630/data/nature/experiments")
RES = BASE / "results" / "cognitive_rsa"
STIM_DIR = BASE / "data" / "cognitive_stimuli" / "rsa"

CONDITIONS = [
    "anger", "belief", "disgust", "empathy", "fear", "happiness",
    "intention", "judgment", "mentalizing", "moral", "sadness",
    "self_referential", "theory_of_mind", "valence",
]


def load_stimuli():
    """Load RSA stimuli with condition labels."""
    stim_path = STIM_DIR / "rsa_stimuli.jsonl"
    if not stim_path.exists():
        stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    if not stim_path.exists():
        # Try to find stimuli elsewhere
        for p in BASE.rglob("rsa_stimuli.jsonl"):
            stim_path = p
            break

    stimuli = []
    with open(stim_path) as f:
        for line in f:
            d = json.loads(line)
            stimuli.append(d)
    return stimuli


def rdm_cosine(act):
    act = act - act.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - np.clip(a @ a.T, -1, 1)


def triu_vec(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]


def load_brain_rdm():
    brain = np.load(RES / "brain_rdm.npz", allow_pickle=True)
    return brain["rdm"], list(brain["conditions"])


def glove_baseline(stimuli, brain_rdm, brain_conds):
    """RSA using GloVe mean-pooled embeddings."""
    print("\n=== GloVe Baseline ===")

    # Try to load GloVe
    glove_path = Path("/hpc2hdd/home/mzhang630/data/glove")
    glove_file = glove_path / "glove.6B.300d.txt"

    if not glove_file.exists():
        # Download GloVe
        print("  Downloading GloVe 6B 300d...")
        import urllib.request, zipfile
        glove_path.mkdir(parents=True, exist_ok=True)
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = glove_path / "glove.6B.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract("glove.6B.300d.txt", glove_path)
        print("  Downloaded.")

    # Load GloVe vectors
    print("  Loading GloVe vectors...")
    glove = {}
    with open(glove_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    print(f"  Loaded {len(glove)} word vectors")

    # Compute mean GloVe embedding per stimulus
    by_cond = defaultdict(list)
    for s in stimuli:
        cond = s.get("condition", s.get("category"))
        text = s.get("text", s.get("prompt", ""))
        words = re.findall(r'\w+', text.lower())
        vecs = [glove[w] for w in words if w in glove]
        if vecs:
            by_cond[cond].append(np.mean(vecs, axis=0))

    # Build condition centroids
    cond_list = [c for c in brain_conds if c in by_cond]
    centroids = np.array([np.mean(by_cond[c], axis=0) for c in cond_list])

    # RDM
    rdm = rdm_cosine(centroids)

    # Compare with brain
    brain_idx = [brain_conds.index(c) for c in cond_list]
    brain_sub = brain_rdm[np.ix_(brain_idx, brain_idx)]
    triu = np.triu_indices(len(cond_list), k=1)

    rho, p = spearmanr(triu_vec(brain_sub), triu_vec(rdm))
    print(f"  GloVe brain-RSA: ρ = {rho:+.4f}, p = {p:.1e}")
    print(f"  (Conditions: {len(cond_list)})")

    return {"method": "glove_300d", "rho": float(rho), "p": float(p),
            "n_conditions": len(cond_list)}


def tfidf_baseline(stimuli, brain_rdm, brain_conds):
    """RSA using TF-IDF bag-of-words features."""
    print("\n=== TF-IDF Baseline ===")

    by_cond = defaultdict(list)
    for s in stimuli:
        cond = s.get("condition", s.get("category"))
        text = s.get("text", s.get("prompt", ""))
        by_cond[cond].append(text)

    cond_list = [c for c in brain_conds if c in by_cond]

    # Concatenate all texts per condition
    cond_texts = [" ".join(by_cond[c]) for c in cond_list]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(cond_texts).toarray()

    # RDM
    rdm = rdm_cosine(tfidf_matrix)

    brain_idx = [brain_conds.index(c) for c in cond_list]
    brain_sub = brain_rdm[np.ix_(brain_idx, brain_idx)]
    triu = np.triu_indices(len(cond_list), k=1)

    rho, p = spearmanr(triu_vec(brain_sub), triu_vec(rdm))
    print(f"  TF-IDF brain-RSA: ρ = {rho:+.4f}, p = {p:.1e}")

    return {"method": "tfidf_5000", "rho": float(rho), "p": float(p),
            "n_conditions": len(cond_list)}


def condition_name_baseline(brain_rdm, brain_conds):
    """RSA using only condition label names as input to a simple embedding."""
    print("\n=== Condition-Name Baseline ===")

    # Use GloVe to embed condition names
    glove_path = Path("/hpc2hdd/home/mzhang630/data/glove/glove.6B.300d.txt")
    glove = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)

    label_map = {
        "anger": ["anger", "angry"],
        "fear": ["fear", "afraid"],
        "disgust": ["disgust"],
        "sadness": ["sadness", "sad"],
        "happiness": ["happiness", "happy"],
        "valence": ["valence", "emotion", "feeling"],
        "belief": ["belief", "believe"],
        "mentalizing": ["mentalizing", "mental", "mind"],
        "intention": ["intention", "intend", "plan"],
        "theory_of_mind": ["theory", "mind", "perspective"],
        "empathy": ["empathy", "compassion"],
        "self_referential": ["self", "referential", "introspection"],
        "judgment": ["judgment", "judge", "evaluate"],
        "moral": ["moral", "ethics", "ethical"],
    }

    centroids = []
    cond_list = []
    for c in brain_conds:
        words = label_map.get(c, [c])
        vecs = [glove[w] for w in words if w in glove]
        if vecs:
            centroids.append(np.mean(vecs, axis=0))
            cond_list.append(c)

    centroids = np.array(centroids)
    rdm = rdm_cosine(centroids)

    brain_idx = [brain_conds.index(c) for c in cond_list]
    brain_sub = brain_rdm[np.ix_(brain_idx, brain_idx)]
    triu = np.triu_indices(len(cond_list), k=1)

    rho, p = spearmanr(triu_vec(brain_sub), triu_vec(rdm))
    print(f"  Condition-name brain-RSA: ρ = {rho:+.4f}, p = {p:.1e}")
    print(f"  (If ρ is high, the LLM result may just reflect concept-name semantics)")

    return {"method": "condition_name_glove", "rho": float(rho), "p": float(p),
            "n_conditions": len(cond_list)}


def length_baseline(stimuli, brain_rdm, brain_conds):
    """RSA using only sentence length as feature."""
    print("\n=== Sentence-Length Baseline ===")

    by_cond = defaultdict(list)
    for s in stimuli:
        cond = s.get("condition", s.get("category"))
        text = s.get("text", s.get("prompt", ""))
        by_cond[cond].append(len(text.split()))

    cond_list = [c for c in brain_conds if c in by_cond]
    means = np.array([[np.mean(by_cond[c])] for c in cond_list])

    # 1D "RDM" = absolute difference in mean length
    n = len(cond_list)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(means[i, 0] - means[j, 0])

    brain_idx = [brain_conds.index(c) for c in cond_list]
    brain_sub = brain_rdm[np.ix_(brain_idx, brain_idx)]
    triu = np.triu_indices(n, k=1)

    rho, p = spearmanr(triu_vec(brain_sub), triu_vec(rdm))
    print(f"  Length brain-RSA: ρ = {rho:+.4f}, p = {p:.1e}")

    # Also report mean length per condition
    print(f"\n  Mean sentence length per condition:")
    for c in cond_list:
        print(f"    {c:<20s} {np.mean(by_cond[c]):>6.1f} words (n={len(by_cond[c])})")

    return {"method": "sentence_length", "rho": float(rho), "p": float(p),
            "n_conditions": len(cond_list)}


def main():
    print("Loading stimuli...")
    stimuli = load_stimuli()
    print(f"  {len(stimuli)} stimuli")

    print("Loading brain RDM...")
    brain_rdm, brain_conds = load_brain_rdm()
    print(f"  {brain_rdm.shape[0]} conditions: {brain_conds}")

    results = []

    # 1. GloVe
    r = glove_baseline(stimuli, brain_rdm, brain_conds)
    results.append(r)

    # 2. TF-IDF
    r = tfidf_baseline(stimuli, brain_rdm, brain_conds)
    results.append(r)

    # 3. Condition name only
    r = condition_name_baseline(brain_rdm, brain_conds)
    results.append(r)

    # 4. Sentence length
    r = length_baseline(stimuli, brain_rdm, brain_conds)
    results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Trained LLM (Qwen 7B):        ρ ≈ +0.64  (for reference)")
    for r in results:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(f"  {r['method']:<30s} ρ = {r['rho']:+.4f}  p = {r['p']:.1e}  {sig}")

    # Save
    with open(RES / "baseline_controls.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RES / 'baseline_controls.json'}")


if __name__ == "__main__":
    main()
