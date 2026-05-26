#!/usr/bin/env python3
"""
Experiment B: Brain-to-LLM transfer.

The brain distinguishes belief/intention/ToM/empathy but LLMs collapse them.
Extract the missing fine-grained structure from brain RDM and test whether
injecting it into LLM representations improves social reasoning benchmarks.

Approach:
1. From brain RDM, extract the within-mentalistic distance structure
2. Find LLM directions that best approximate these brain distances
3. During inference on ToM tasks, amplify these directions
4. Measure accuracy on ToM/social reasoning benchmarks

Benchmarks:
- False belief stories (Sally-Anne type)
- Faux pas detection
- Intention vs outcome distinction in moral judgment

Usage:
  python brain_transfer.py \
    --model_path /path/to/Qwen2.5-3B-Instruct \
    --brain_rdm /path/to/narratives_brain_rdm.npz \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

FALSE_BELIEF_TASKS = [
    {
        "story": "Sally puts her ball in the basket and leaves the room. While she is gone, Anne moves the ball to the box. Sally comes back.",
        "question": "Where will Sally look for her ball?",
        "correct": "basket",
        "foil": "box",
    },
    {
        "story": "John puts chocolate in the cupboard and goes to school. His mother moves the chocolate to the drawer. John comes home hungry.",
        "question": "Where will John look for the chocolate?",
        "correct": "cupboard",
        "foil": "drawer",
    },
    {
        "story": "Mary tells Peter she is going to the library. But actually she goes to the cinema. Later, Lisa asks Peter where Mary is.",
        "question": "What will Peter tell Lisa?",
        "correct": "library",
        "foil": "cinema",
    },
    {
        "story": "Tom watches his friend hide a toy under the pillow. Tom leaves. His friend moves the toy to the closet. Tom returns to play.",
        "question": "Where will Tom look for the toy first?",
        "correct": "pillow",
        "foil": "closet",
    },
    {
        "story": "Emma puts her keys on the table before showering. Her roommate moves the keys to the hook by the door. Emma finishes showering.",
        "question": "Where will Emma look for her keys?",
        "correct": "table",
        "foil": "hook",
    },
]

FAUX_PAS_TASKS = [
    {
        "story": "Jill had just moved to a new town. She went to a party where she met a woman named Helen. They talked about how they had both moved from other cities. A few weeks later, Jill ran into Helen at the grocery store. Helen said, 'Nice to meet you! Are you new in town?'",
        "question": "Did someone say something they shouldn't have?",
        "correct": "yes",
        "explanation": "Helen forgot she had already met Jill",
    },
    {
        "story": "Mike bought his friend Steve a crystal bowl for a wedding present. Steve had his wedding and a few months later, Mike was over at Steve's for dinner. Mike accidentally knocked the bowl off the table and it shattered. 'Don't worry,' said Steve, 'I never liked it anyway. Someone gave it to me for my wedding.'",
        "question": "Did someone say something they shouldn't have?",
        "correct": "yes",
        "explanation": "Steve didn't realize Mike was the one who gave the bowl",
    },
    {
        "story": "A teacher was planning a surprise birthday party for one of the students. She told all the other students to keep it a secret. On the day of the party, one student said to the birthday child, 'I hope you like chocolate cake!'",
        "question": "Did someone say something they shouldn't have?",
        "correct": "yes",
        "explanation": "The student accidentally revealed information about the surprise",
    },
]

INTENTION_VS_OUTCOME_TASKS = [
    {
        "scenario": "Person A tries to poison Person B's coffee but accidentally grabs sugar instead. Person B is fine.",
        "question": "How morally wrong is Person A's action? (1=not wrong, 7=very wrong)",
        "expected_high": True,
        "reasoning": "Bad intention despite good outcome",
    },
    {
        "scenario": "Person A tries to add sugar to Person B's coffee but accidentally grabs poison. Person B gets sick.",
        "question": "How morally wrong is Person A's action? (1=not wrong, 7=very wrong)",
        "expected_high": False,
        "reasoning": "Good intention despite bad outcome",
    },
    {
        "scenario": "A driver swerves to avoid a cat and accidentally hits a pedestrian who suffers minor injuries.",
        "question": "How morally wrong is the driver's action? (1=not wrong, 7=very wrong)",
        "expected_high": False,
        "reasoning": "Good intention, bad outcome",
    },
    {
        "scenario": "A driver deliberately tries to scare a pedestrian by driving close but the pedestrian doesn't notice.",
        "question": "How morally wrong is the driver's action? (1=not wrong, 7=very wrong)",
        "expected_high": True,
        "reasoning": "Bad intention, no bad outcome",
    },
]


def get_mentalistic_directions(model, tokenizer, brain_rdm, brain_conds, device):
    """Extract directions that separate mentalistic sub-conditions based on brain RDM."""
    ment_conds = ["belief", "mentalizing", "intention", "theory_of_mind", "empathy"]
    ment_texts = {
        "belief": [
            "She was convinced he had already left the building.",
            "He believed the story she told him without question.",
            "They assumed the project was cancelled based on the email.",
        ],
        "mentalizing": [
            "He tried to figure out what she was really thinking.",
            "She could tell from his expression that something was bothering him.",
            "The therapist worked to understand the patient's state of mind.",
        ],
        "intention": [
            "She planned to reveal the truth at the meeting.",
            "He deliberately chose not to mention the problem.",
            "The company intended to expand into new markets next year.",
        ],
        "theory_of_mind": [
            "He didn't realize she already knew about the surprise party.",
            "The child couldn't understand that others might see things differently.",
            "She knew he was lying but pretended to believe him.",
        ],
        "empathy": [
            "She felt his pain as if it were her own.",
            "The nurse was deeply moved by the patient's struggle.",
            "He couldn't help but share in her joy at the good news.",
        ],
    }

    cond_acts = {}
    for cond in ment_conds:
        acts = []
        for text in ment_texts[cond]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            last = out.hidden_states[-1]
            mask = inputs["attention_mask"].bool()
            mean_act = last[0][mask[0]].mean(dim=0)
            acts.append(mean_act)
        cond_acts[cond] = torch.stack(acts).mean(dim=0)

    # Extract brain-derived directions between mentalistic conditions
    directions = {}
    ment_in_brain = [c for c in ment_conds if c in brain_conds]
    for i, c1 in enumerate(ment_in_brain):
        for c2 in ment_in_brain[i+1:]:
            if c1 in cond_acts and c2 in cond_acts:
                bi, bj = brain_conds.index(c1), brain_conds.index(c2)
                brain_dist = brain_rdm[bi, bj]
                llm_dir = cond_acts[c2] - cond_acts[c1]
                llm_dir = llm_dir / llm_dir.norm()
                directions[f"{c1}->{c2}"] = {
                    "direction": llm_dir,
                    "brain_distance": float(brain_dist),
                }
    return directions


def evaluate_false_belief(model, tokenizer, device, steering_hook=None):
    """Evaluate false belief understanding with optional steering."""
    correct = 0
    for task in FALSE_BELIEF_TASKS:
        prompt = f"{task['story']}\n\nQuestion: {task['question']}\nAnswer briefly:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        hooks = []
        if steering_hook:
            hooks = steering_hook(model)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
        for h in hooks:
            h.remove()

        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True).lower()

        if task["correct"].lower() in response and task["foil"].lower() not in response:
            correct += 1
        elif task["correct"].lower() in response:
            correct += 0.5

    return correct / len(FALSE_BELIEF_TASKS)


def evaluate_faux_pas(model, tokenizer, device, steering_hook=None):
    """Evaluate faux pas detection."""
    correct = 0
    for task in FAUX_PAS_TASKS:
        prompt = f"{task['story']}\n\nQuestion: {task['question']}\nAnswer yes or no, then explain briefly:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        hooks = []
        if steering_hook:
            hooks = steering_hook(model)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
        for h in hooks:
            h.remove()

        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True).lower()

        if task["correct"].lower() in response[:20]:
            correct += 1

    return correct / len(FAUX_PAS_TASKS)


def make_steering_hook(model, direction, alpha):
    """Create hooks that steer activations along a direction."""
    def apply_hooks(model):
        hooks = []
        n_layers = model.config.num_hidden_layers
        for layer_idx in range(n_layers * 3 // 4, n_layers):
            def hook_fn(module, input, output, d=direction, a=alpha):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h = h + a * d.to(h.device).unsqueeze(0).unsqueeze(0)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            hooks.append(model.model.layers[layer_idx].register_forward_hook(hook_fn))
        return hooks
    return apply_hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--brain_rdm", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    brain_data = np.load(args.brain_rdm, allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])

    print("Extracting mentalistic directions...")
    directions = get_mentalistic_directions(model, tokenizer, brain_rdm, brain_conds, device)
    for name, d in directions.items():
        print(f"  {name}: brain_dist={d['brain_distance']:.3f}")

    # Baseline evaluation
    print("\n=== Baseline (no steering) ===")
    fb_base = evaluate_false_belief(model, tokenizer, device)
    fp_base = evaluate_faux_pas(model, tokenizer, device)
    print(f"  False Belief accuracy: {fb_base:.1%}")
    print(f"  Faux Pas detection:    {fp_base:.1%}")

    # Steering with belief->ToM direction (amplify ToM understanding)
    results = {"model": args.model_short, "baseline": {"false_belief": fb_base, "faux_pas": fp_base}}
    results["steering"] = []

    # Try different directions and alphas
    for dir_name, dir_info in directions.items():
        if "theory_of_mind" not in dir_name:
            continue
        direction = dir_info["direction"]

        for alpha in [5, 10, 20, 50]:
            hook_fn = make_steering_hook(model, direction, alpha)
            fb = evaluate_false_belief(model, tokenizer, device, hook_fn)
            fp = evaluate_faux_pas(model, tokenizer, device, hook_fn)

            print(f"\n  Steering {dir_name} alpha={alpha}:")
            print(f"    False Belief: {fb:.1%} (base: {fb_base:.1%}, Δ={fb-fb_base:+.1%})")
            print(f"    Faux Pas:     {fp:.1%} (base: {fp_base:.1%}, Δ={fp-fp_base:+.1%})")

            results["steering"].append({
                "direction": dir_name,
                "alpha": alpha,
                "false_belief": fb,
                "faux_pas": fp,
                "delta_fb": fb - fb_base,
                "delta_fp": fp - fp_base,
            })

    out_path = Path(args.output_dir) / f"{args.model_short}_brain_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
