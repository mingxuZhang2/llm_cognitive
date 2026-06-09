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
import json, os, re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "cognitive_rsa"
STIM_DIR = BASE / "data" / "cognitive_stimuli" / "rsa"
GLOVE_DIR = Path(os.environ.get("GLOVE_DIR", str(BASE / "data" / "glove")))

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
    glove_path = GLOVE_DIR
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
    glove_path = GLOVE_DIR / "glove.6B.300d.txt"
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


def partial_spearman_rsa(target, model_vec, nuisance_vecs):
    """Partial Spearman RSA: correlate target and model after regressing out nuisances."""
    from scipy.stats import rankdata
    from numpy.linalg import lstsq

    t_rank = rankdata(target)
    m_rank = rankdata(model_vec)
    n_ranks = np.column_stack([rankdata(nv) for nv in nuisance_vecs])
    X = np.column_stack([n_ranks, np.ones(len(t_rank))])

    beta_t, _, _, _ = lstsq(X, t_rank, rcond=None)
    t_resid = t_rank - X @ beta_t

    beta_m, _, _, _ = lstsq(X, m_rank, rcond=None)
    m_resid = m_rank - X @ beta_m

    rho = np.corrcoef(t_resid, m_resid)[0, 1]

    rng = np.random.default_rng(42)
    count = 0
    n_perm = 5000
    for _ in range(n_perm):
        perm = rng.permutation(len(m_resid))
        if np.corrcoef(t_resid, m_resid[perm])[0, 1] >= rho:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return rho, p


def partial_rsa_analysis(stimuli, brain_rdm, brain_conds):
    """Run partial RSA controlling for multiple nuisance RDMs."""
    print("\n=== Partial RSA: Trained LLM controlling for confounds ===")

    n = len(brain_conds)
    triu_idx = np.triu_indices(n, k=1)
    brain_vec = brain_rdm[triu_idx]
    by_cond = defaultdict(list)
    for s in stimuli:
        c = s.get("condition", s.get("category"))
        t = s.get("text", s.get("prompt", ""))
        by_cond[c].append(t)

    # Build nuisance RDMs
    # Length RDM
    length_means = np.array([np.mean([len(t.split()) for t in by_cond[c]]) for c in brain_conds])
    length_rdm = np.abs(length_means[:, None] - length_means[None, :])
    length_vec = length_rdm[triu_idx]

    # GloVe RDM
    glove = {}
    glove_path = GLOVE_DIR / "glove.6B.300d.txt"
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)

    glove_centroids = []
    for c in brain_conds:
        vecs = []
        for t in by_cond[c]:
            words = re.findall(r'\w+', t.lower())
            wv = [glove[w] for w in words if w in glove]
            if wv:
                vecs.append(np.mean(wv, axis=0))
        glove_centroids.append(np.mean(vecs, axis=0) if vecs else np.zeros(300))
    gc = np.array(glove_centroids)
    gc = gc - gc.mean(axis=0)
    gn = np.linalg.norm(gc, axis=1, keepdims=True); gn[gn==0]=1
    glove_rdm = 1.0 - np.clip((gc/gn) @ (gc/gn).T, -1, 1)
    glove_vec = glove_rdm[triu_idx]

    # Condition-name RDM
    label_map = {
        "anger": ["anger","angry"], "fear": ["fear","afraid"],
        "disgust": ["disgust"], "sadness": ["sadness","sad"],
        "happiness": ["happiness","happy"], "valence": ["valence","emotion","feeling"],
        "belief": ["belief","believe"], "mentalizing": ["mentalizing","mental","mind"],
        "intention": ["intention","intend","plan"],
        "theory_of_mind": ["theory","mind","perspective"],
        "empathy": ["empathy","compassion"],
        "self_referential": ["self","referential","introspection"],
        "judgment": ["judgment","judge","evaluate"],
        "moral": ["moral","ethics","ethical"],
    }
    name_cents = []
    for c in brain_conds:
        ws = label_map.get(c, [c])
        vecs = [glove[w] for w in ws if w in glove]
        name_cents.append(np.mean(vecs, axis=0) if vecs else np.zeros(300))
    nc = np.array(name_cents)
    nc = nc - nc.mean(axis=0)
    nn = np.linalg.norm(nc, axis=1, keepdims=True); nn[nn==0]=1
    name_rdm = 1.0 - np.clip((nc/nn) @ (nc/nn).T, -1, 1)
    name_vec = name_rdm[triu_idx]

    # Load trained LLM RDM
    llm_npz = RES / "Qwen2.5-7B-Instruct_rsa_v2_per_stim.npz"
    data = np.load(llm_npz, allow_pickle=True)
    per_stim = data["per_stim_activations"]
    conditions = list(data["conditions"])
    pool_idx = list(data["pooling_names"]).index("mean_all")
    unique_conds = sorted(set(conditions))
    stim_cond = np.array([unique_conds.index(c) for c in conditions])
    n_layers = per_stim.shape[2]
    cond_means = np.zeros((len(unique_conds), n_layers, per_stim.shape[-1]), dtype=np.float32)
    for ci in range(len(unique_conds)):
        idx = np.where(stim_cond == ci)[0]
        cond_means[ci] = per_stim[pool_idx, idx].mean(axis=0)
    order = [unique_conds.index(c) for c in brain_conds]
    cond_means = cond_means[order]

    best_rho, best_L = -1, 0
    for L in range(n_layers):
        act = cond_means[:, L, :].astype(np.float64)
        act = act - act.mean(axis=0)
        nn2 = np.linalg.norm(act, axis=1, keepdims=True); nn2[nn2==0]=1
        rdm = 1.0 - np.clip((act/nn2) @ (act/nn2).T, -1, 1)
        rho, _ = spearmanr(brain_vec, rdm[triu_idx])
        if rho > best_rho:
            best_rho, best_L = rho, L
    act = cond_means[:, best_L, :].astype(np.float64)
    act = act - act.mean(axis=0)
    nn2 = np.linalg.norm(act, axis=1, keepdims=True); nn2[nn2==0]=1
    llm_rdm = 1.0 - np.clip((act/nn2) @ (act/nn2).T, -1, 1)
    llm_vec = llm_rdm[triu_idx]
    del per_stim, data

    rho_raw, _ = spearmanr(brain_vec, llm_vec)
    print(f"  Raw LLM-brain:                    ρ = {rho_raw:+.4f}")

    rho_len, p_len = partial_spearman_rsa(brain_vec, llm_vec, [length_vec])
    print(f"  LLM | length:                     ρ = {rho_len:+.4f}  p = {p_len:.4f}")

    rho_glv, p_glv = partial_spearman_rsa(brain_vec, llm_vec, [glove_vec])
    print(f"  LLM | GloVe:                      ρ = {rho_glv:+.4f}  p = {p_glv:.4f}")

    rho_name, p_name = partial_spearman_rsa(brain_vec, llm_vec, [name_vec])
    print(f"  LLM | condition-name:             ρ = {rho_name:+.4f}  p = {p_name:.4f}")

    rho_all, p_all = partial_spearman_rsa(brain_vec, llm_vec,
                                          [length_vec, glove_vec, name_vec])
    print(f"  LLM | length+GloVe+name:          ρ = {rho_all:+.4f}  p = {p_all:.4f}")

    print(f"\n  Raw: {rho_raw:+.4f} → after all controls: {rho_all:+.4f} "
          f"({rho_all/rho_raw*100:.0f}% retained)")

    return {
        "raw_rho": float(rho_raw),
        "partial_length": {"rho": float(rho_len), "p": float(p_len)},
        "partial_glove": {"rho": float(rho_glv), "p": float(p_glv)},
        "partial_name": {"rho": float(rho_name), "p": float(p_name)},
        "partial_all": {"rho": float(rho_all), "p": float(p_all)},
        "pct_retained": float(rho_all / rho_raw * 100),
    }


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

    # 5. Partial RSA: trained LLM controlling for nuisance RDMs
    partial_results = partial_rsa_analysis(stimuli, brain_rdm, brain_conds)
    results.append({"method": "partial_rsa", **partial_results})

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Trained LLM (Qwen 7B):        ρ ≈ +0.64  (for reference)")
    for r in results:
        if r["method"] == "partial_rsa":
            continue
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(f"  {r['method']:<30s} ρ = {r['rho']:+.4f}  p = {r['p']:.1e}  {sig}")

    # Save
    with open(RES / "baseline_controls.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RES / 'baseline_controls.json'}")


if __name__ == "__main__":
    main()
