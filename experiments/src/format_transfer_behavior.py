#!/usr/bin/env python3
"""
Behavioral format transfer test: do LLMs correctly interpret social-cognitive
content regardless of surface format?

For each of the 4 social conditions x 4 formats x 15 items, append a diagnostic
question and measure accuracy via logprob comparison (forced two-choice).

Diagnostic questions test whether the model understands the CONTENT:
  false_belief  — "Where does the person think the object is?" (original vs moved)
  intention     — "What is the person most likely trying to do?" (true vs surface)
  moral_judgment — "Is the described action ethically acceptable?" (yes vs no)
  self_referential — "What does this person recognize about themselves?" (insight vs opposite)

If format does not affect comprehension, accuracy should be constant across formats.
If it does, behavioral degradation in non-canonical formats suggests the RSA
signal was partly carried by format-specific processing.

GPU required. One model per run.
Usage:
  python src/format_transfer_behavior.py \
      --model_path /path/to/model \
      --model_short Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
STIM_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "content_format_cross_stimuli.jsonl"
OUT_DIR = BASE / "results" / "format_transfer_behavior"

CONDITIONS = ["false_belief", "intention", "moral_judgment", "self_referential"]
FORMATS = ["narrative", "dialogue", "record", "list"]

PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}

# ══════════════════════════════════════════════════════════════════════
# Diagnostic questions per condition
#
# Each entry: (question_template, correct_answer, wrong_answer)
# The question is appended after the stimulus. We measure:
#   log P(correct_answer) vs log P(wrong_answer)
#
# For false_belief: the "correct" answer is the ORIGINAL location
#   (where the person THINKS the item is — the false belief)
# For intention: the correct answer reflects the true underlying goal
# For moral_judgment: we measure consistency of judgment across formats
#   (correct = the majority-model answer; we check FORMAT INVARIANCE)
# For self_referential: the correct answer matches the introspective insight
# ══════════════════════════════════════════════════════════════════════

# Generic QA for false_belief and intention — must work across ALL formats
# (narrative, dialogue, record, list) since each format has different scenarios.
# The question tests whether the model grasps the cognitive function regardless
# of surface presentation.

FALSE_BELIEF_QA = [
    ("Does the main person in this passage have an accurate understanding of the current situation?",
     "No, they are mistaken about how things actually are",
     "Yes, they have a correct understanding of the situation"),
] * 15  # same generic question for all 15 items per format

INTENTION_QA = [
    ("Based on the clues described, is the person's underlying goal explicitly stated or must it be inferred?",
     "It must be inferred from indirect evidence",
     "It is directly and explicitly stated"),
] * 15  # same generic question for all 15 items per format

MORAL_JUDGMENT_QA = [
    # For moral judgment, we test whether the model recognizes the ethical tension.
    # Correct = acknowledges the dilemma; Wrong = sees no issue.
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
    ("Does this situation involve a genuine ethical conflict?", "Yes, there are competing values at stake", "No, the right choice is obvious"),
]

SELF_REFERENTIAL_QA = [
    ("What is this person recognizing about themselves?", "A gap between their public persona and private self", "Complete satisfaction with who they are"),
    ("What is this person recognizing about themselves?", "That they have changed significantly over time", "That they have always been the same person"),
    ("What is this person recognizing about themselves?", "That a criticism exposed a hidden vulnerability", "That other people's opinions do not affect them"),
    ("What is this person recognizing about themselves?", "A pattern of avoidance in their life choices", "A clear sense of purpose and direction"),
    ("What is this person recognizing about themselves?", "That their caution may mask a deeper avoidance", "That they are simply a careful and prudent person"),
    ("What is this person recognizing about themselves?", "How much they have changed from their younger self", "That they have remained true to their original values"),
    ("What is this person recognizing about themselves?", "That perfectionism may stem from fear of inadequacy", "That high standards are purely about quality"),
    ("What is this person recognizing about themselves?", "That they consistently chose security over passion", "That they always followed their dreams"),
    ("What is this person recognizing about themselves?", "A pattern of creating distance in relationships", "That they are naturally independent and comfortable alone"),
    ("What is this person recognizing about themselves?", "That the stress comes from within, not from the job", "That the job itself is the sole source of stress"),
    ("What is this person recognizing about themselves?", "That much of their daily life involves performance", "That they are always genuine and authentic"),
    ("What is this person recognizing about themselves?", "That their generosity may be driven by need", "That their generosity is purely selfless"),
    ("What is this person recognizing about themselves?", "A disconnect between resume facts and inner identity", "That external achievements fully define who they are"),
    ("What is this person recognizing about themselves?", "Unresolved guilt that persists over time", "That past issues have been fully resolved"),
    ("What is this person recognizing about themselves?", "Uncomfortable truths about their personality", "Perfect alignment between self-image and reality"),
]

CONDITION_QA = {
    "false_belief": FALSE_BELIEF_QA,
    "intention": INTENTION_QA,
    "moral_judgment": MORAL_JUDGMENT_QA,
    "self_referential": SELF_REFERENTIAL_QA,
}


def load_stimuli() -> list[dict]:
    """Load the 240 crossed stimuli."""
    if not STIM_PATH.exists():
        raise FileNotFoundError(
            f"Stimuli not found at {STIM_PATH}. "
            "Run content_format_cross_stimuli.py first."
        )
    with open(STIM_PATH) as f:
        stimuli = [json.loads(line) for line in f]
    print(f"Loaded {len(stimuli)} stimuli from {STIM_PATH.name}")
    return stimuli


def get_answer_logprob(model, tokenizer, prompt, answer, device):
    """Compute log probability of an answer continuation given a prompt."""
    # Encode prompt + answer together
    full_text = prompt + " " + answer
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)

    # The answer tokens are the ones beyond the prompt
    answer_start = len(prompt_ids)
    if answer_start >= len(full_ids):
        # Fallback: just use the last few tokens
        answer_start = max(0, len(full_ids) - 5)

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0]  # (seq_len, vocab)

    # Sum log probs of answer tokens
    log_probs = torch.log_softmax(logits, dim=-1)
    total_lp = 0.0
    n_tokens = 0
    for pos in range(answer_start, len(full_ids)):
        if pos > 0:  # log prob of token at position pos given prefix
            token_id = full_ids[pos]
            lp = log_probs[pos - 1, token_id].item()
            total_lp += lp
            n_tokens += 1

    # Return mean log prob (length-normalized)
    mean_lp = total_lp / max(n_tokens, 1)
    return mean_lp, total_lp, n_tokens


def evaluate_stimulus(model, tokenizer, stimulus, item_within_cond, device):
    """Evaluate one stimulus with its condition-specific diagnostic question."""
    cond = stimulus["condition"]
    qa_list = CONDITION_QA[cond]

    # Each condition has 15 items, and item_within_cond indexes them
    question, correct, wrong = qa_list[item_within_cond]

    prompt = stimulus["text"] + "\n\nQuestion: " + question + "\nAnswer:"

    lp_correct_mean, lp_correct_sum, n_correct = get_answer_logprob(
        model, tokenizer, prompt, correct, device)
    lp_wrong_mean, lp_wrong_sum, n_wrong = get_answer_logprob(
        model, tokenizer, prompt, wrong, device)

    is_correct = lp_correct_mean > lp_wrong_mean
    confidence = lp_correct_mean - lp_wrong_mean

    return {
        "condition": cond,
        "format": stimulus["format"],
        "item_id": stimulus["item_id"],
        "item_within_cond": item_within_cond,
        "question": question,
        "correct_answer": correct,
        "wrong_answer": wrong,
        "lp_correct_mean": float(lp_correct_mean),
        "lp_wrong_mean": float(lp_wrong_mean),
        "lp_correct_sum": float(lp_correct_sum),
        "lp_wrong_sum": float(lp_wrong_sum),
        "is_correct": bool(is_correct),
        "confidence": float(confidence),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Behavioral format transfer test via logprob")
    ap.add_argument("--model_path", required=True, help="Path to HuggingFace model")
    ap.add_argument("--model_short", default=None, help="Short model name")
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"FORMAT TRANSFER BEHAVIORAL TEST: {model_short}")
    print("=" * 70)
    print(f"Device: {device}")

    t0 = time.time()

    # ── Load stimuli ──
    stimuli = load_stimuli()

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ── Evaluate all stimuli ──
    print(f"\nEvaluating {len(stimuli)} stimuli...")
    results = []

    # Track item index within each condition
    # Structure: for each condition, items appear in order across 4 formats
    # (format 0: items 0-14, format 1: items 0-14, etc.)
    cond_item_counter = defaultdict(int)

    for si, stim in enumerate(stimuli):
        cond = stim["condition"]
        fmt = stim["format"]

        # Determine which item (0-14) within this condition
        # Items repeat for each format, so item_within_cond cycles 0-14
        item_within_cond = cond_item_counter[cond] % 15
        cond_item_counter[cond] += 1

        result = evaluate_stimulus(
            model, tokenizer, stim, item_within_cond, device)
        results.append(result)

        if (si + 1) % 20 == 0:
            recent = results[-20:]
            acc = sum(r["is_correct"] for r in recent) / len(recent)
            print(f"  [{si+1:3d}/{len(stimuli)}] "
                  f"recent accuracy: {acc:.2f}")

    # Free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # Per-cell accuracy
    cell_results = defaultdict(list)
    for r in results:
        cell_results[(r["condition"], r["format"])].append(r["is_correct"])

    print(f"\nAccuracy by condition x format:")
    header = f"{'':20s}" + "".join(f"{f:>12s}" for f in FORMATS) + f"{'MEAN':>8s}"
    print(header)
    print("-" * len(header))

    cond_accs = defaultdict(list)
    fmt_accs = defaultdict(list)
    cell_acc_dict = {}

    for cond in CONDITIONS:
        row = f"{cond:20s}"
        for fmt in FORMATS:
            vals = cell_results[(cond, fmt)]
            acc = sum(vals) / len(vals) if vals else 0.0
            cell_acc_dict[(cond, fmt)] = acc
            cond_accs[cond].append(acc)
            fmt_accs[fmt].append(acc)
            row += f"{acc:12.3f}"
        mean_cond = np.mean(cond_accs[cond])
        row += f"{mean_cond:8.3f}"
        print(row)

    # Mean by format
    print(f"\n{'MEAN':20s}", end="")
    for fmt in FORMATS:
        mean_fmt = np.mean(fmt_accs[fmt])
        print(f"{mean_fmt:12.3f}", end="")
    overall = sum(r["is_correct"] for r in results) / len(results)
    print(f"{overall:8.3f}")

    # Overall by condition
    print(f"\nOverall by condition (collapsing formats):")
    for cond in CONDITIONS:
        vals = [r["is_correct"] for r in results if r["condition"] == cond]
        acc = sum(vals) / len(vals) if vals else 0.0
        print(f"  {cond:20s}: {acc:.3f} ({sum(vals)}/{len(vals)})")

    # Overall by format
    print(f"\nOverall by format (collapsing conditions):")
    for fmt in FORMATS:
        vals = [r["is_correct"] for r in results if r["format"] == fmt]
        acc = sum(vals) / len(vals) if vals else 0.0
        print(f"  {fmt:20s}: {acc:.3f} ({sum(vals)}/{len(vals)})")

    # Canonical vs non-canonical
    # "Canonical" format for each condition (the most natural pairing):
    canonical = {
        "false_belief": "narrative",   # false belief stories are naturally narrative
        "intention": "narrative",      # intention inference is naturally narrative
        "moral_judgment": "dialogue",  # moral dilemmas often come as conversations
        "self_referential": "list",    # self-reflection often in list/journal form
    }

    canonical_accs = []
    noncanonical_accs = []
    for cond in CONDITIONS:
        canon_fmt = canonical[cond]
        for fmt in FORMATS:
            acc = cell_acc_dict[(cond, fmt)]
            if fmt == canon_fmt:
                canonical_accs.append(acc)
            else:
                noncanonical_accs.append(acc)

    mean_canonical = np.mean(canonical_accs)
    mean_noncanonical = np.mean(noncanonical_accs)
    print(f"\nCanonical vs Non-canonical format:")
    print(f"  Canonical format accuracy:     {mean_canonical:.3f}")
    print(f"  Non-canonical format accuracy:  {mean_noncanonical:.3f}")
    print(f"  Delta: {mean_canonical - mean_noncanonical:+.3f}")

    # Format consistency: std of accuracy across formats per condition
    print(f"\nFormat consistency (std across formats per condition):")
    for cond in CONDITIONS:
        accs = [cell_acc_dict[(cond, fmt)] for fmt in FORMATS]
        std = np.std(accs)
        print(f"  {cond:20s}: std={std:.3f} "
              f"(range: {min(accs):.3f}-{max(accs):.3f})")

    # Mean confidence by cell
    print(f"\nMean confidence (lp_correct - lp_wrong) by condition x format:")
    header2 = f"{'':20s}" + "".join(f"{f:>12s}" for f in FORMATS)
    print(header2)
    for cond in CONDITIONS:
        row = f"{cond:20s}"
        for fmt in FORMATS:
            confs = [r["confidence"] for r in results
                     if r["condition"] == cond and r["format"] == fmt]
            mc = np.mean(confs) if confs else 0.0
            row += f"{mc:+12.3f}"
        print(row)

    # ══════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": model_short,
        "n_stimuli": len(stimuli),
        "overall_accuracy": round(float(overall), 4),
        "accuracy_by_condition": {
            cond: round(float(np.mean(cond_accs[cond])), 4) for cond in CONDITIONS
        },
        "accuracy_by_format": {
            fmt: round(float(np.mean(fmt_accs[fmt])), 4) for fmt in FORMATS
        },
        "accuracy_by_cell": {
            f"{cond}/{fmt}": round(float(cell_acc_dict[(cond, fmt)]), 4)
            for cond in CONDITIONS for fmt in FORMATS
        },
        "canonical_vs_noncanonical": {
            "canonical_accuracy": round(float(mean_canonical), 4),
            "noncanonical_accuracy": round(float(mean_noncanonical), 4),
            "delta": round(float(mean_canonical - mean_noncanonical), 4),
        },
        "format_consistency_std": {
            cond: round(float(np.std([cell_acc_dict[(cond, fmt)]
                                      for fmt in FORMATS])), 4)
            for cond in CONDITIONS
        },
        "per_stimulus_results": results,
    }

    out_path = OUT_DIR / f"{model_short}_format_transfer.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
