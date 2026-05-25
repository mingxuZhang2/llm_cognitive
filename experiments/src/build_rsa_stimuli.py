"""
Build the RSA stimulus set: condition-tagged jsonl covering 14 cognitive
conditions, each paired to a brain map (Neurosynth or HCP). Each stimulus is
normalized to a single ``text`` field so the same forward-pass extractor can
process all conditions uniformly.

Output:
    cognitive_stimuli/rsa/rsa_stimuli.jsonl
        records: {id, condition, text, source}
    cognitive_stimuli/rsa/rsa_conditions_manifest.json
        per-condition metadata: brain_map path, stimulus source files, counts
"""

from __future__ import annotations

import json
import random
from pathlib import Path


BASE = Path("/hpc2hdd/home/mzhang630/data/nature/experiments/data/cognitive_stimuli")
BRAIN_DIR = Path("/hpc2hdd/home/mzhang630/data/nature/experiments/data/brain_maps")
OUT_DIR = BASE / "rsa"
OUT_STIM = OUT_DIR / "rsa_stimuli.jsonl"
OUT_MANIFEST = OUT_DIR / "rsa_conditions_manifest.json"

SEED = 20260525
TARGET_PER_COND = 60  # cap; some conditions have fewer items naturally


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def sample(items, n, rng):
    if len(items) <= n:
        return items[:]
    return rng.sample(items, n)


def build():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    manifest = {
        "seed": SEED,
        "target_per_condition": TARGET_PER_COND,
        "conditions": {},
    }

    # Pre-load reusable sources
    mfv = load_jsonl(BASE / "moral" / "moral_foundations_vignettes.jsonl")
    mfv_care = [d for d in mfv if d.get("foundation") == "care"]
    mfv_non_care = [d for d in mfv if d.get("foundation") != "care"]
    moral_dil = load_jsonl(BASE / "moral" / "moral_dilemmas.jsonl")
    ethics = load_jsonl(BASE / "moral" / "ethics_sample.jsonl")
    indirect = load_jsonl(BASE / "tom" / "indirect_requests.jsonl")
    false_belief = load_jsonl(BASE / "tom" / "false_belief.jsonl")
    strange = load_jsonl(BASE / "tom" / "strange_stories.jsonl")
    faux_pas = load_jsonl(BASE / "tom" / "faux_pas.jsonl")
    self_other = load_jsonl(BASE / "tom" / "self_other.jsonl")
    so_self = [d for d in self_other if d.get("perspective") == "self"]
    so_other = [d for d in self_other if d.get("perspective") == "other"]
    emo_loc = load_jsonl(BASE / "emotion" / "emotion_localizer_stimuli.jsonl")
    emo_by = {e: [d for d in emo_loc if d.get("emotion") == e]
              for e in ("anger", "fear", "sadness", "disgust", "joy")}
    vad = load_jsonl(BASE / "emotion" / "vad_graded_sentences.jsonl")

    def add_cond(cond, brain_rel, stim_records, source_tag):
        chosen = sample(stim_records, TARGET_PER_COND, rng)
        for i, d in enumerate(chosen):
            records.append({
                "id": f"rsa_{cond}_{i:03d}",
                "condition": cond,
                "text": d["__text__"],
                "source": source_tag,
                "src_id": d.get("__src_id__", ""),
            })
        brain_abs = BRAIN_DIR / brain_rel
        assert brain_abs.exists(), f"missing brain map: {brain_abs}"
        manifest["conditions"][cond] = {
            "brain_map": brain_rel,
            "source": source_tag,
            "n_total_pool": len(stim_records),
            "n_selected": len(chosen),
        }

    # ---- condition assembly ----

    # 1. moral: moral dilemmas + MFV non-care vignettes
    moral_pool = []
    for d in moral_dil:
        moral_pool.append({"__text__": d["scenario"], "__src_id__": d["id"]})
    for d in mfv_non_care:
        moral_pool.append({"__text__": d["vignette_text"], "__src_id__": d["id"]})
    add_cond("moral", "neurosynth/moral.nii.gz", moral_pool,
             "moral_dilemmas + MFV non-care")

    # 2. judgment: ethics_sample
    judg_pool = [{"__text__": d["text"], "__src_id__": d["id"]} for d in ethics]
    add_cond("judgment", "neurosynth/judgment.nii.gz", judg_pool,
             "ETHICS commonsense")

    # 3. intention: indirect_requests (context + utterance)
    intent_pool = [{"__text__": f"{d['context']} {d['utterance']}",
                    "__src_id__": d["id"]} for d in indirect]
    add_cond("intention", "neurosynth/intention.nii.gz", intent_pool,
             "indirect_requests")

    # 4. belief: false_belief story + question
    belief_pool = [{"__text__": f"{d['story']} {d['question']}",
                    "__src_id__": d["id"]} for d in false_belief]
    add_cond("belief", "neurosynth/beliefs.nii.gz", belief_pool,
             "false_belief")

    # 5. mentalizing: strange_stories (vignette + question)
    ment_pool = [{"__text__": f"{d['story']} {d['question']}",
                  "__src_id__": d["id"]} for d in strange]
    add_cond("mentalizing", "neurosynth/mentalizing.nii.gz", ment_pool,
             "strange_stories")

    # 6. theory_of_mind: self_other other-perspective + faux_pas stories
    tom_pool = []
    for d in so_other:
        tom_pool.append({"__text__": f"{d['scenario']} {d['question']}",
                         "__src_id__": d["id"]})
    for d in faux_pas:
        tom_pool.append({"__text__": d["story"], "__src_id__": d["id"]})
    add_cond("theory_of_mind", "hcp/social_tom_vs_random.nii.gz", tom_pool,
             "self_other(other) + faux_pas")

    # 7. empathy: MFV care vignettes
    emp_pool = [{"__text__": d["vignette_text"], "__src_id__": d["id"]}
                for d in mfv_care]
    add_cond("empathy", "neurosynth/empathy.nii.gz", emp_pool,
             "MFV care foundation")

    # 8. self_referential: self_other self-perspective
    self_pool = [{"__text__": f"{d['scenario']} {d['question']}",
                  "__src_id__": d["id"]} for d in so_self]
    add_cond("self_referential", "neurosynth/self_referential.nii.gz",
             self_pool, "self_other(self)")

    # 9-13. basic emotions: emotion_localizer
    for cond, ekman, brain_slug in [
        ("anger",     "anger",   "anger"),
        ("fear",      "fear",    "fear"),
        ("sadness",   "sadness", "sadness"),
        ("disgust",   "disgust", "disgust"),
        ("happiness", "joy",     "happiness"),
    ]:
        pool = [{"__text__": d["text"], "__src_id__": d["id"]}
                for d in emo_by[ekman]]
        add_cond(cond, f"neurosynth/{brain_slug}.nii.gz", pool,
                 f"emotion_localizer({ekman})")

    # 14. valence: vad_graded_sentences
    val_pool = [{"__text__": d["text"], "__src_id__": d["id"]} for d in vad]
    add_cond("valence", "neurosynth/valence.nii.gz", val_pool,
             "vad_graded_sentences")

    # Write outputs
    with open(OUT_STIM, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(records)} stimuli across {len(manifest['conditions'])} "
          f"conditions to {OUT_STIM}")
    for cond, info in manifest["conditions"].items():
        print(f"  {cond:>16s}: n={info['n_selected']:<3d}  brain={info['brain_map']}")
    print(f"Manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    build()
