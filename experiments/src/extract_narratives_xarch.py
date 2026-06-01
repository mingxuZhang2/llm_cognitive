#!/usr/bin/env python3
"""
Multi-story Narratives LLM extraction for the CROSS-ARCHITECTURE defense.

Loads a model ONCE and extracts per-sentence, per-layer mean-pooled activations
for all 12 target stories (the same stories the Narratives brain RDM aggregates,
see narratives_brain_rdm.py), saving {model_short}_{story}_narratives.npz per story
in the SAME format as extract_narratives_llm.py (so the existing tooling + the
cross-arch compare both work).

Purpose: the headline RSA uses Neurosynth meta-analytic maps, which are derived from
the *text* of ~14k fMRI papers. A reviewer can argue a text model matching them is
text-matching-text. Narratives is the clean answer: real subjects, real BOLD, the SAME
story text fed to brain and model. Extending it from Qwen-only to all 4 architectures is
the load-bearing defense of the core finding.

GPU job -> scripts/slurm/narratives_xarch_extract.sh.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_narratives_llm import extract   # reuse the exact per-layer mean-pool logic

# the 12 stories the Narratives brain RDM aggregates over (narratives_brain_rdm.TARGET_STORIES)
TARGET_STORIES = ["pieman", "tunnel", "notthefallintact", "black", "prettymouth",
                  "forgot", "sherlock", "merlin", "bronx", "21styear",
                  "slumlordreach", "lucy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--sentences_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    all_sents = [json.loads(l) for l in open(args.sentences_jsonl)]
    by_story = defaultdict(list)
    for s in all_sents:
        if s["story"] in TARGET_STORIES:
            by_story[s["story"]].append(s)
    print(f"target stories present: { {k: len(v) for k, v in by_story.items()} }", flush=True)

    print(f"Loading {args.model_path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    print(f"device {device}", flush=True)

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    for story in TARGET_STORIES:
        sents = by_story.get(story, [])
        if not sents:
            print(f"[skip] {story}: 0 sentences", flush=True); continue
        texts = [s["text"] for s in sents]
        labels = [s.get("llm_labels", []) for s in sents]
        onsets = [s["onset"] for s in sents]
        offsets = [s["offset"] for s in sents]
        acts = extract(model, tok, texts, str(device), args.batch_size)
        out = outdir / f"{args.model_short}_{story}_narratives.npz"
        np.savez_compressed(out, activations=acts, texts=np.array(texts, dtype=object),
                            labels=np.array(labels, dtype=object),
                            onsets=np.array(onsets), offsets=np.array(offsets),
                            story=story, model_short=args.model_short)
        print(f"  saved {out.name}  acts={acts.shape}", flush=True)
    print(f"ALL STORIES DONE for {args.model_short}", flush=True)


if __name__ == "__main__":
    main()
