#!/usr/bin/env python3
"""
ToM format sensitivity: does social cognition performance degrade with format change?

Standard false belief tasks (narrative format) → high accuracy.
Same logical content in compressed/restructured format → accuracy drops?

This parallels our RSA finding: standard stimuli → within-social ρ=0.55;
template-matched (uniform format) → ρ=0.12.

Three task types × two formats × 10 scenarios = 60 items per model.
GPU required (text generation + answer extraction).

Usage:
  python src/tom_format_sensitivity.py --model_path /path/to/model [--model_short name]
"""
from __future__ import annotations
import argparse, json, re, gc
import numpy as np
import torch
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# FALSE BELIEF TASKS (10 scenarios × 2 formats)
# ═══════════════════════════════════════════════════════════════════════

FALSE_BELIEF_STANDARD = [
    {"story": "Mia puts her cookies in the cookie jar and goes outside to water the plants. While Mia is outside, her brother moves the cookies from the cookie jar to the cupboard above the stove. Mia comes back inside.",
     "question": "Where will Mia look for her cookies?",
     "correct": "cookie jar", "wrong": "cupboard"},
    {"story": "Sophia puts her wallet in her purse and goes to wash her hands. While Sophia is washing her hands, her roommate takes the wallet from the purse and puts it in the desk drawer. Sophia returns.",
     "question": "Where will Sophia look for her wallet?",
     "correct": "purse", "wrong": "desk drawer"},
    {"story": "James hides his birthday present for his wife in the closet before leaving for work. While James is at work, his wife finds the present and moves it under the bed. James comes home.",
     "question": "Where will James look for the birthday present?",
     "correct": "closet", "wrong": "under the bed"},
    {"story": "A child puts their favorite toy in the red box and goes to school. While the child is at school, their mother moves the toy from the red box to the blue box. The child comes home.",
     "question": "Where will the child look for the toy?",
     "correct": "red box", "wrong": "blue box"},
    {"story": "Tom places his lunch in the refrigerator and goes to a meeting. During the meeting, his coworker accidentally takes Tom's lunch and puts it in the office cabinet. Tom returns from the meeting.",
     "question": "Where will Tom look for his lunch?",
     "correct": "refrigerator", "wrong": "office cabinet"},
    {"story": "Elena leaves her keys on the kitchen table and goes upstairs to shower. While she is showering, her husband moves the keys from the table to the hook by the front door. Elena comes downstairs.",
     "question": "Where will Elena look for her keys?",
     "correct": "kitchen table", "wrong": "hook by the front door"},
    {"story": "A boy puts his book on the shelf in his room and goes out to play. His sister comes in and moves the book from the shelf to the drawer. The boy comes back to his room.",
     "question": "Where will the boy look for his book?",
     "correct": "shelf", "wrong": "drawer"},
    {"story": "Maria stores her phone in her jacket pocket and hangs the jacket in the hallway. Her son takes the phone from the jacket pocket and puts it on the living room couch. Maria goes to get her jacket.",
     "question": "Where will Maria look for her phone?",
     "correct": "jacket pocket", "wrong": "living room couch"},
    {"story": "David puts the car keys in the bowl by the entrance and leaves to walk the dog. His wife moves the keys from the bowl to her handbag because she needs the car. David returns from walking the dog.",
     "question": "Where will David look for the car keys?",
     "correct": "bowl by the entrance", "wrong": "handbag"},
    {"story": "A teacher puts the exam papers in the top drawer of her desk and goes to lunch. The janitor, while cleaning, moves the papers to the bottom drawer. The teacher returns from lunch.",
     "question": "Where will the teacher look for the exam papers?",
     "correct": "top drawer", "wrong": "bottom drawer"},
]

FALSE_BELIEF_COMPRESSED = [
    {"story": "Mia thinks her cookies are in the cookie jar. Actually, her brother moved them to the cupboard while she was outside.",
     "question": "Where will Mia look for her cookies?",
     "correct": "cookie jar", "wrong": "cupboard"},
    {"story": "Sophia thinks her wallet is in her purse. Actually, her roommate moved it to the desk drawer while she was away.",
     "question": "Where will Sophia look for her wallet?",
     "correct": "purse", "wrong": "desk drawer"},
    {"story": "James thinks the birthday present is in the closet. Actually, his wife moved it under the bed while he was at work.",
     "question": "Where will James look for the present?",
     "correct": "closet", "wrong": "under the bed"},
    {"story": "The child thinks the toy is in the red box. Actually, their mother moved it to the blue box while they were at school.",
     "question": "Where will the child look for the toy?",
     "correct": "red box", "wrong": "blue box"},
    {"story": "Tom thinks his lunch is in the refrigerator. Actually, his coworker moved it to the office cabinet during the meeting.",
     "question": "Where will Tom look for his lunch?",
     "correct": "refrigerator", "wrong": "office cabinet"},
    {"story": "Elena thinks her keys are on the kitchen table. Actually, her husband moved them to the hook by the front door while she was upstairs.",
     "question": "Where will Elena look for her keys?",
     "correct": "kitchen table", "wrong": "hook by the front door"},
    {"story": "The boy thinks his book is on the shelf. Actually, his sister moved it to the drawer while he was out playing.",
     "question": "Where will the boy look for his book?",
     "correct": "shelf", "wrong": "drawer"},
    {"story": "Maria thinks her phone is in her jacket pocket. Actually, her son moved it to the living room couch.",
     "question": "Where will Maria look for her phone?",
     "correct": "jacket pocket", "wrong": "living room couch"},
    {"story": "David thinks the car keys are in the bowl by the entrance. Actually, his wife moved them to her handbag.",
     "question": "Where will David look for the car keys?",
     "correct": "bowl by the entrance", "wrong": "handbag"},
    {"story": "The teacher thinks the exam papers are in the top drawer. Actually, the janitor moved them to the bottom drawer during lunch.",
     "question": "Where will the teacher look for the exam papers?",
     "correct": "top drawer", "wrong": "bottom drawer"},
]

# ═══════════════════════════════════════════════════════════════════════
# INTENTION READING (10 scenarios × 2 formats)
# ═══════════════════════════════════════════════════════════════════════

INTENTION_STANDARD = [
    {"story": "Two people are sitting in a living room. The window is open and a cold breeze is coming in. One person stands up, walks over to the window, and says: 'It's really freezing in here, isn't it?'",
     "question": "What does the person want?",
     "correct": "close the window", "wrong": "talk about the weather"},
    {"story": "A mother walks into her son's room. Clothes are scattered all over the floor and the bed is unmade. She looks at the mess, sighs, and says: 'What a lovely room you have here.'",
     "question": "What does the mother want?",
     "correct": "clean the room", "wrong": "compliment the room"},
    {"story": "At a restaurant, a customer has been waiting for 40 minutes. When the waiter finally comes by, the customer says: 'Oh, no rush at all. I have all day.'",
     "question": "What does the customer want?",
     "correct": "be served faster", "wrong": "continue waiting"},
    {"story": "A student is struggling with a math problem. She turns to her classmate who is good at math and says: 'This problem is impossible. I wonder if anyone could solve it.'",
     "question": "What does the student want?",
     "correct": "help with the problem", "wrong": "express frustration"},
    {"story": "A man is carrying several heavy grocery bags. He approaches the front door of his house where his teenager is sitting on the porch playing on their phone. He says: 'Boy, these bags are really heavy.'",
     "question": "What does the man want?",
     "correct": "help carrying the bags", "wrong": "comment on the weight"},
    {"story": "At a meeting, the boss looks at the clock, then looks at the team, and says: 'Well, I think we've covered everything we need to, haven't we?'",
     "question": "What does the boss want?",
     "correct": "end the meeting", "wrong": "confirm the agenda"},
    {"story": "A woman at a party is standing near the exit with her coat on. She says to her friend: 'Well, it's been a lovely evening. I have an early morning tomorrow.'",
     "question": "What does the woman want?",
     "correct": "leave the party", "wrong": "describe her schedule"},
    {"story": "A roommate looks at the overflowing trash can in the kitchen. She turns to her other roommate who is watching TV and says: 'Have you noticed the trash lately?'",
     "question": "What does the roommate want?",
     "correct": "take out the trash", "wrong": "discuss the trash"},
    {"story": "A passenger in a car notices the driver is going 90 in a 60 zone. The passenger says: 'Wow, we're really making good time, aren't we?'",
     "question": "What does the passenger want?",
     "correct": "slow down", "wrong": "praise the speed"},
    {"story": "A teacher hands back a test to a student who got a low grade. The teacher says: 'I'm sure you'll find the material much easier if you spend more time on it.'",
     "question": "What does the teacher want?",
     "correct": "study more", "wrong": "feel better about the grade"},
]

INTENTION_COMPRESSED = [
    {"story": "Someone feels cold because the window is open. They say: 'It's really freezing in here, isn't it?'",
     "question": "What do they want?",
     "correct": "close the window", "wrong": "talk about the weather"},
    {"story": "A mother sees her son's messy room. She says sarcastically: 'What a lovely room you have here.'",
     "question": "What does she want?",
     "correct": "clean the room", "wrong": "compliment the room"},
    {"story": "A customer waited 40 minutes for service. They say sarcastically: 'Oh, no rush at all. I have all day.'",
     "question": "What do they want?",
     "correct": "be served faster", "wrong": "continue waiting"},
    {"story": "A student can't solve a math problem. She tells a smart classmate: 'This problem is impossible. I wonder if anyone could solve it.'",
     "question": "What does she want?",
     "correct": "help with the problem", "wrong": "express frustration"},
    {"story": "A man carrying heavy bags says to his idle teenager: 'Boy, these bags are really heavy.'",
     "question": "What does he want?",
     "correct": "help carrying the bags", "wrong": "comment on the weight"},
    {"story": "The boss looks at the clock and says: 'Well, I think we've covered everything, haven't we?'",
     "question": "What does the boss want?",
     "correct": "end the meeting", "wrong": "confirm the agenda"},
    {"story": "A woman with her coat on stands near the exit and says: 'Well, it's been lovely. I have an early morning.'",
     "question": "What does she want?",
     "correct": "leave the party", "wrong": "describe her schedule"},
    {"story": "A roommate looks at the overflowing trash and says to the other: 'Have you noticed the trash lately?'",
     "question": "What does she want?",
     "correct": "take out the trash", "wrong": "discuss the trash"},
    {"story": "A passenger in a speeding car says: 'Wow, we're really making good time, aren't we?'",
     "question": "What do they want?",
     "correct": "slow down", "wrong": "praise the speed"},
    {"story": "A teacher gives back a bad test grade and says: 'You'll find it easier if you spend more time on it.'",
     "question": "What does the teacher want?",
     "correct": "study more", "wrong": "feel better about the grade"},
]

PROMPT_TEMPLATE = """{story}

{question}

Answer in one short phrase. Do not explain."""


def generate(model, tokenizer, prompt, device, max_new=50):
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, temperature=0.01,
            top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def score_answer(response, correct, wrong):
    resp = response.lower().strip().rstrip(".")
    correct_l = correct.lower()
    wrong_l = wrong.lower()
    if correct_l in resp and wrong_l not in resp:
        return 1
    if wrong_l in resp and correct_l not in resp:
        return 0
    if correct_l in resp and wrong_l in resp:
        ci = resp.index(correct_l)
        wi = resp.index(wrong_l)
        return 1 if ci < wi else 0
    return -1  # ambiguous


def run_tasks(model, tokenizer, device, tasks, label):
    correct = 0
    total = 0
    ambiguous = 0
    details = []
    for i, task in enumerate(tasks):
        prompt = PROMPT_TEMPLATE.format(story=task["story"], question=task["question"])
        resp = generate(model, tokenizer, prompt, device)
        s = score_answer(resp, task["correct"], task["wrong"])
        if s >= 0:
            correct += s
            total += 1
        else:
            ambiguous += 1
        details.append({"response": resp, "score": s, "correct_ans": task["correct"]})
    acc = correct / total if total > 0 else 0
    print(f"    {label}: {correct}/{total} correct ({acc:.1%}), {ambiguous} ambiguous")
    return {"accuracy": float(acc), "correct": correct, "total": total,
            "ambiguous": ambiguous, "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*70}")
    print(f"ToM FORMAT SENSITIVITY: {model_short}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    results = {"model": model_short}

    print(f"\n  --- FALSE BELIEF (n=10) ---")
    results["false_belief_standard"] = run_tasks(
        model, tokenizer, device, FALSE_BELIEF_STANDARD, "Standard (narrative)")
    results["false_belief_compressed"] = run_tasks(
        model, tokenizer, device, FALSE_BELIEF_COMPRESSED, "Compressed (minimal)")

    print(f"\n  --- INTENTION READING (n=10) ---")
    results["intention_standard"] = run_tasks(
        model, tokenizer, device, INTENTION_STANDARD, "Standard (dialogue)")
    results["intention_compressed"] = run_tasks(
        model, tokenizer, device, INTENTION_COMPRESSED, "Compressed (minimal)")

    del model; gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_short}")
    print(f"{'='*70}")
    print(f"  {'Task':>25s} | {'Standard':>10s} | {'Compressed':>10s} | {'Δ':>8s}")
    print(f"  {'-'*60}")

    for task in ["false_belief", "intention"]:
        std_acc = results[f"{task}_standard"]["accuracy"]
        cmp_acc = results[f"{task}_compressed"]["accuracy"]
        delta = cmp_acc - std_acc
        print(f"  {task:>25s} | {std_acc:>9.1%} | {cmp_acc:>9.1%} | {delta:>+7.1%}")

    overall_std = np.mean([results[f"{t}_standard"]["accuracy"] for t in ["false_belief", "intention"]])
    overall_cmp = np.mean([results[f"{t}_compressed"]["accuracy"] for t in ["false_belief", "intention"]])
    print(f"  {'OVERALL':>25s} | {overall_std:>9.1%} | {overall_cmp:>9.1%} | {overall_cmp-overall_std:>+7.1%}")

    out_path = OUT / f"tom_format_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
