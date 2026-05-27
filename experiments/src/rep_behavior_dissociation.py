#!/usr/bin/env python3
"""
Representation-Behavior Dissociation: the killer experiment.

Tests whether behavioral Theory-of-Mind (ToM) performance and representational
Mentalistic Separability Index (MSI) are DISSOCIATED across model scale and
prompting conditions.

Key prediction: larger models and chain-of-thought (CoT) prompting IMPROVE
behavioral accuracy on social-cognition tasks but DO NOT change MSI — the
model's internal representational geometry stays collapsed regardless of
behavioral competence.

Part A — Mentalistic Separability Index (MSI):
  Load per-stimulus RSA activations from existing NPZ files. Compute 14
  condition centroids at the peak layer, build cosine-distance RDM, then:
    MSI = mean(within_mentalistic_distances) / mean(within_affective_distances)
  Brain MSI from brain_rdm.npz provides the reference.

Part B — ToM Behavioral Benchmark:
  80 forced-choice (A/B) items across 8 social-cognition skills (10 each),
  evaluated under Standard and CoT prompting.

Part C — Output:
  Single JSON per model with MSI, brain MSI, RDM, per-item behavioural
  results, and summary accuracy.

Usage (HPC3):
  python rep_behavior_dissociation.py \
    --model_path /data/user/mzhang630/data/models/Qwen2.5-7B-Instruct \
    --model_short Qwen2.5-7B-Instruct \
    --rsa_dir results/cognitive_rsa \
    --brain_rdm_path results/cognitive_rsa/brain_rdm.npz \
    --output_dir results/rep_behavior_dissociation
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ── Condition taxonomy ────────────────────────────────────────────────
CONDITIONS = [
    "anger", "belief", "disgust", "empathy", "fear", "happiness",
    "intention", "judgment", "mentalizing", "moral", "sadness",
    "self_referential", "theory_of_mind", "valence",
]

AFFECTIVE = {"anger", "disgust", "fear", "happiness", "sadness", "valence"}
MENTALISTIC = {"belief", "empathy", "intention", "mentalizing",
               "self_referential", "theory_of_mind"}
BOUNDARY = {"judgment", "moral"}

# ── Helper functions ──────────────────────────────────────────────────

def rdm_cosine(act: np.ndarray) -> np.ndarray:
    """Cosine-distance RDM from centred activations [n, d]."""
    act = act - act.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - np.clip(a @ a.T, -1, 1)


def triu_vec(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices(mat.shape[0], k=1)]


def pair_indices(conds: list[str], subset: set[str]):
    """Return upper-triangle (i,j) pairs where both i,j belong to *subset*."""
    idx = [i for i, c in enumerate(conds) if c in subset]
    pairs = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs.append((idx[a], idx[b]))
    return pairs


def mean_rdm_subset(rdm: np.ndarray, conds: list[str],
                    subset: set[str]) -> float:
    pairs = pair_indices(conds, subset)
    return float(np.mean([rdm[i, j] for i, j in pairs]))


# =====================================================================
#  PART A: Mentalistic Separability Index
# =====================================================================

def compute_msi(rsa_dir: Path, model_short: str, brain_rdm_path: Path,
                peak_layer: int, model_path: str | None = None,
                stimuli_path: str | None = None):
    """Return dict with MSI, brain_MSI, RDM, peak_layer."""
    npz_path = rsa_dir / f"{model_short}_rsa_v2_per_stim.npz"

    if npz_path.exists():
        print(f"  Loading existing RSA activations: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        per_stim = data["per_stim_activations"]  # [3, 712, layers, dim]
        conditions = list(data["conditions"])
        pooling_names = list(data["pooling_names"])
        pool_idx = pooling_names.index("mean_all")
        n_layers = per_stim.shape[2]
        if peak_layer >= n_layers:
            print(f"  Warning: peak_layer {peak_layer} >= n_layers {n_layers}; "
                  f"clamping to {n_layers - 1}")
            peak_layer = n_layers - 1

        # Condition centroids at peak layer
        unique_conds = sorted(set(conditions))
        centroids = np.zeros((len(unique_conds), per_stim.shape[-1]),
                             dtype=np.float64)
        for ci, c in enumerate(unique_conds):
            mask = np.array([cond == c for cond in conditions])
            centroids[ci] = per_stim[pool_idx, mask, peak_layer, :].mean(axis=0)

        # Reorder to canonical CONDITIONS order
        order = [unique_conds.index(c) for c in CONDITIONS]
        centroids = centroids[order]
    else:
        # Extract activations from scratch
        print(f"  No existing NPZ found at {npz_path}.")
        if model_path is None or stimuli_path is None:
            raise FileNotFoundError(
                f"RSA NPZ not found and --model_path / --stimuli_path not "
                f"provided. Cannot extract activations.")
        centroids, peak_layer = _extract_centroids(
            model_path, stimuli_path, peak_layer)

    # Cosine-distance RDM (centroids already ordered as CONDITIONS)
    rdm = rdm_cosine(centroids)

    msi_ment = mean_rdm_subset(rdm, CONDITIONS, MENTALISTIC)
    msi_aff = mean_rdm_subset(rdm, CONDITIONS, AFFECTIVE)
    msi = msi_ment / msi_aff if msi_aff > 0 else float("nan")

    # Brain MSI
    brain_data = np.load(brain_rdm_path, allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])
    brain_msi_ment = mean_rdm_subset(brain_rdm, brain_conds, MENTALISTIC)
    brain_msi_aff = mean_rdm_subset(brain_rdm, brain_conds, AFFECTIVE)
    brain_msi = brain_msi_ment / brain_msi_aff if brain_msi_aff > 0 else float("nan")

    # Sub-RDM alignment: Spearman correlation of within-group distance vectors
    def _sub_rdm_corr(rdm_a, conds_a, rdm_b, conds_b, subset):
        pairs_a = pair_indices(conds_a, subset)
        pairs_b = pair_indices(conds_b, subset)
        vec_a = [rdm_a[i, j] for i, j in pairs_a]
        vec_b = [rdm_b[i, j] for i, j in pairs_b]
        if len(vec_a) < 3:
            return float("nan"), float("nan")
        rho, p = spearmanr(vec_a, vec_b)
        return float(rho), float(p)

    ment_align_rho, ment_align_p = _sub_rdm_corr(
        rdm, CONDITIONS, brain_rdm, brain_conds, MENTALISTIC)
    aff_align_rho, aff_align_p = _sub_rdm_corr(
        rdm, CONDITIONS, brain_rdm, brain_conds, AFFECTIVE)

    # Mean absolute residual per group
    def _mean_residual(rdm_a, conds_a, rdm_b, conds_b, subset):
        pairs_a = pair_indices(conds_a, subset)
        pairs_b = pair_indices(conds_b, subset)
        return float(np.mean([abs(rdm_a[ia, ja] - rdm_b[ib, jb])
                              for (ia, ja), (ib, jb) in zip(pairs_a, pairs_b)]))

    ment_residual = _mean_residual(rdm, CONDITIONS, brain_rdm, brain_conds, MENTALISTIC)
    aff_residual = _mean_residual(rdm, CONDITIONS, brain_rdm, brain_conds, AFFECTIVE)

    print(f"\n  MSI  (model)  = {msi:.4f}  "
          f"(ment={msi_ment:.4f}, aff={msi_aff:.4f})")
    print(f"  MSI  (brain)  = {brain_msi:.4f}  "
          f"(ment={brain_msi_ment:.4f}, aff={brain_msi_aff:.4f})")
    print(f"  Mentalistic alignment:  ρ={ment_align_rho:+.4f}  p={ment_align_p:.4f}")
    print(f"  Affective alignment:    ρ={aff_align_rho:+.4f}  p={aff_align_p:.4f}")
    print(f"  Mentalistic residual:   {ment_residual:.4f}")
    print(f"  Affective residual:     {aff_residual:.4f}")

    return {
        "msi": msi,
        "msi_mentalistic": msi_ment,
        "msi_affective": msi_aff,
        "brain_msi": brain_msi,
        "brain_msi_mentalistic": brain_msi_ment,
        "brain_msi_affective": brain_msi_aff,
        "mentalistic_alignment_rho": ment_align_rho,
        "mentalistic_alignment_p": ment_align_p,
        "affective_alignment_rho": aff_align_rho,
        "affective_alignment_p": aff_align_p,
        "mentalistic_residual": ment_residual,
        "affective_residual": aff_residual,
        "peak_layer": peak_layer,
        "rdm": rdm.tolist(),
    }


def _extract_centroids(model_path: str, stimuli_path: str,
                       peak_layer: int):
    """Fallback: load model, extract mean-pooled activations, return centroids."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model for activation extraction: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device

    stimuli = []
    with open(stimuli_path) as f:
        for line in f:
            line = line.strip()
            if line:
                stimuli.append(json.loads(line))

    conditions_per_stim = [s["condition"] for s in stimuli]
    unique_conds = sorted(set(conditions_per_stim))
    n_conds = len(unique_conds)

    # Accumulate per-condition activations
    cond_sums = {}
    cond_counts = {}
    for c in unique_conds:
        cond_counts[c] = 0

    print(f"  Extracting activations from {len(stimuli)} stimuli ...")
    for si, stim in enumerate(stimuli):
        text = stim["text"]
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512, add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        # Mean-pooled hidden state at peak layer
        hs = out.hidden_states[peak_layer]  # [1, seq, dim]
        mask = enc["attention_mask"][0].bool()
        mean_act = hs[0][mask].float().mean(dim=0).cpu().numpy()
        c = stim["condition"]
        if c not in cond_sums:
            cond_sums[c] = np.zeros_like(mean_act, dtype=np.float64)
        cond_sums[c] += mean_act
        cond_counts[c] += 1
        if (si + 1) % 100 == 0:
            print(f"    {si+1}/{len(stimuli)}")

    centroids = np.zeros((n_conds, len(cond_sums[unique_conds[0]])),
                         dtype=np.float64)
    for ci, c in enumerate(unique_conds):
        centroids[ci] = cond_sums[c] / cond_counts[c]

    # Reorder to canonical CONDITIONS
    order = [unique_conds.index(c) for c in CONDITIONS]
    centroids = centroids[order]
    return centroids, peak_layer


# =====================================================================
#  PART B: ToM Behavioural Benchmark — 80 A/B Forced-Choice Items
# =====================================================================

TOM_ITEMS = [
    # ── false_belief (10) ──────────────────────────────────────────
    {
        "skill": "false_belief", "difficulty": "easy",
        "scenario": "Sara puts her chocolate in the blue cupboard and leaves the kitchen. While she is away, her brother Tom moves the chocolate to the red cupboard. Sara comes back and wants her chocolate.",
        "question": "Where will Sara look first for her chocolate?\nA) The blue cupboard\nB) The red cupboard",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "easy",
        "scenario": "Emma tells Jake that the meeting is at 3 PM. After Emma leaves, the boss sends Jake an email changing the meeting to 4 PM. Emma did not see the email.",
        "question": "What time does Emma think the meeting is?\nA) 3 PM\nB) 4 PM",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "easy",
        "scenario": "A child watches a puppet show. Puppet A hides a ball under Box 1 and leaves. Puppet B moves the ball to Box 2. Puppet A returns.",
        "question": "Where will Puppet A look for the ball?\nA) Box 1\nB) Box 2",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "hard",
        "scenario": "Maria tells her roommate Lily that she left the car keys on the hall table. Later, the cleaning lady moves the keys to the drawer in the bedroom. Then Lily, who saw the cleaning lady move the keys, texts Maria: 'I put your keys somewhere safe.' Maria assumes Lily means the hall table drawer next to where she left them.",
        "question": "Where does Maria believe the keys are now?\nA) The hall table drawer\nB) The bedroom drawer",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "hard",
        "scenario": "Carlos bought a surprise gift and hid it in the garage. His wife Diana found it while cleaning but pretended she hadn't seen it. Carlos later moved the gift to the attic because he worried Diana might find it in the garage.",
        "question": "Where does Diana think the gift is?\nA) The garage\nB) The attic",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "hard",
        "scenario": "At a magic show, the magician secretly switches a red ball for a blue ball inside a hat. The audience saw the switch, but the volunteer on stage did not. The magician asks the volunteer to predict what color ball is in the hat.",
        "question": "What color will the volunteer say?\nA) Red\nB) Blue",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "easy",
        "scenario": "Noah leaves his sandwich on the picnic table and goes to get a drink. While he's gone, a dog grabs the sandwich and runs away. Noah's friend Mia, who is sitting at the table, does not tell Noah what happened when he returns.",
        "question": "When Noah returns, where does he think his sandwich is?\nA) On the picnic table\nB) Taken by the dog",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "hard",
        "scenario": "Professor Lin puts the exam answer key in File Cabinet A before leaving for lunch. The teaching assistant moves it to File Cabinet B and emails Professor Lin about the change. However, the email goes to Professor Lin's spam folder and she never reads it.",
        "question": "When Professor Lin returns, where will she look for the answer key?\nA) File Cabinet A\nB) File Cabinet B",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "hard",
        "scenario": "Raj tells his colleague Priya that he parked in Lot C. Actually, Raj misremembered — he parked in Lot D. Priya saw Raj's car in Lot D on her way in but hasn't told Raj yet.",
        "question": "Where does Raj believe his car is parked?\nA) Lot C\nB) Lot D",
        "correct": "A",
    },
    {
        "skill": "false_belief", "difficulty": "easy",
        "scenario": "Lisa watches her mom put cookies in the jar on the counter. Lisa goes to school. While Lisa is at school, her dad eats all the cookies and puts crackers in the jar instead.",
        "question": "When Lisa gets home, what does she expect to find in the jar?\nA) Cookies\nB) Crackers",
        "correct": "A",
    },

    # ── second_order_belief (10) ───────────────────────────────────
    {
        "skill": "second_order_belief", "difficulty": "easy",
        "scenario": "Amy knows that Ben's surprise party is on Saturday. Ben thinks the party is on Sunday because that's what he was told. Amy doesn't know that Ben was given wrong information.",
        "question": "What does Amy think Ben believes about when the party is?\nA) Amy thinks Ben believes it's on Saturday\nB) Amy thinks Ben believes it's on Sunday",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "easy",
        "scenario": "Tom saw that the ice cream shop is closed today. Jerry doesn't know this and is planning to go. Tom hasn't told Jerry yet.",
        "question": "What does Tom think Jerry believes about the ice cream shop?\nA) Tom thinks Jerry believes the shop is open\nB) Tom thinks Jerry believes the shop is closed",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "hard",
        "scenario": "Clara tells David she will be at the library. David tells Eve that Clara is at the library. But actually Clara changed her mind and went to the café instead, and she texted only David about the change. David forgot to update Eve.",
        "question": "What does David think Eve believes about where Clara is?\nA) David thinks Eve believes Clara is at the library\nB) David thinks Eve believes Clara is at the café",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "hard",
        "scenario": "Frank overheard Grace say she hates chocolate. Grace was actually joking, but Frank took it seriously. Later, Hana asks Frank what Grace would want as a dessert gift.",
        "question": "What does Frank think Grace wants?\nA) Frank thinks Grace does not want chocolate\nB) Frank thinks Grace wants chocolate",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "easy",
        "scenario": "Kevin told Laura that the test is easy. Laura told Monica that Kevin said the test is easy. Monica has not taken the test yet.",
        "question": "What does Laura think Monica believes about the test difficulty?\nA) Laura thinks Monica believes the test is easy\nB) Laura thinks Monica has no idea about the test",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "hard",
        "scenario": "Nadia secretly reads Oliver's diary, which says he plans to move to Paris. Oliver later decides to stay but hasn't updated his diary. Nadia doesn't know about the change of plans.",
        "question": "What does Nadia think Oliver believes he will do?\nA) Nadia thinks Oliver believes he will move to Paris\nB) Nadia thinks Oliver believes he will stay",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "hard",
        "scenario": "Peter tells Quinn that Rachel is pregnant. Quinn congratulates Rachel, who is confused because she is not pregnant — Peter was mistaken. Quinn doesn't realize Peter was wrong.",
        "question": "What does Quinn think Rachel knows about her own pregnancy?\nA) Quinn thinks Rachel knows she is pregnant\nB) Quinn thinks Rachel doesn't know yet",
        "correct": "B",
    },
    {
        "skill": "second_order_belief", "difficulty": "easy",
        "scenario": "Steve hides a toy in the box. Tina watches but Steve doesn't see Tina watching. Steve thinks his hiding spot is secret.",
        "question": "What does Steve think Tina knows about the toy location?\nA) Steve thinks Tina doesn't know where the toy is\nB) Steve thinks Tina knows where the toy is",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "hard",
        "scenario": "Uma and Victor are playing a card game. Uma peeks and sees Victor has a king. Victor catches Uma peeking but pretends not to notice. Uma thinks she peeked undetected.",
        "question": "What does Victor think Uma believes about whether she was caught?\nA) Victor thinks Uma believes she was not caught\nB) Victor thinks Uma believes she was caught",
        "correct": "A",
    },
    {
        "skill": "second_order_belief", "difficulty": "easy",
        "scenario": "Wendy tells Xander that the store closes at 9 PM. Actually it closes at 8 PM, but Wendy genuinely believes it's 9 PM. Yara heard Wendy tell Xander.",
        "question": "What does Yara think Xander believes about the closing time?\nA) Yara thinks Xander believes the store closes at 9 PM\nB) Yara thinks Xander believes the store closes at 8 PM",
        "correct": "A",
    },

    # ── intention_attribution (10) ─────────────────────────────────
    {
        "skill": "intention_attribution", "difficulty": "easy",
        "scenario": "Marcus sees a woman trip on the sidewalk. He rushes over and helps her up, asking if she's okay.",
        "question": "Why did Marcus help the woman?\nA) He wanted to help because he saw she was hurt\nB) He bumped into her and was trying to cover it up",
        "correct": "A",
    },
    {
        "skill": "intention_attribution", "difficulty": "easy",
        "scenario": "A child accidentally knocks over a glass of milk while reaching for a cookie across the table.",
        "question": "Why did the glass fall?\nA) The child deliberately pushed the glass\nB) The child accidentally knocked it while reaching for something else",
        "correct": "B",
    },
    {
        "skill": "intention_attribution", "difficulty": "hard",
        "scenario": "During a soccer game, a player slide-tackles an opponent and gets the ball. The opponent falls and gets injured. The referee has to decide whether it was a fair tackle or a foul.",
        "question": "What was the player's primary intention?\nA) To get the ball through a legitimate tackle\nB) To deliberately injure the opponent",
        "correct": "A",
    },
    {
        "skill": "intention_attribution", "difficulty": "hard",
        "scenario": "Angela gives her coworker Brian a very difficult assignment, saying it will be a 'great learning opportunity.' Brian struggles for weeks. Later, Brian learns Angela is up for the same promotion.",
        "question": "Why did Angela give Brian the difficult assignment?\nA) She genuinely wanted Brian to grow professionally\nB) She wanted Brian to fail so she could get the promotion",
        "correct": "B",
    },
    {
        "skill": "intention_attribution", "difficulty": "easy",
        "scenario": "A dog knocks over a vase while excitedly wagging its tail when the owner returns home.",
        "question": "Why did the dog knock over the vase?\nA) The dog was excited and accidentally hit it\nB) The dog intentionally broke the vase out of anger",
        "correct": "A",
    },
    {
        "skill": "intention_attribution", "difficulty": "hard",
        "scenario": "During a board meeting, Chen presents data that makes the marketing department look bad. Chen is from the finance department. The data is accurate but Chen chose to highlight the worst quarter specifically.",
        "question": "What was Chen's likely intention?\nA) To provide an unbiased financial overview\nB) To strategically undermine the marketing department",
        "correct": "B",
    },
    {
        "skill": "intention_attribution", "difficulty": "easy",
        "scenario": "A parent packs an extra sandwich in their child's lunchbox along with a note saying 'Have a great day!'",
        "question": "Why did the parent pack the extra sandwich?\nA) Out of care and wanting the child to have enough food\nB) To make other children jealous",
        "correct": "A",
    },
    {
        "skill": "intention_attribution", "difficulty": "hard",
        "scenario": "A politician visits a flooded town for two hours, takes many photos with victims, promises aid, and leaves. Three months later no aid has arrived.",
        "question": "What was the politician's main purpose for the visit?\nA) To assess the damage and coordinate real relief efforts\nB) To gain public sympathy and media coverage for political benefit",
        "correct": "B",
    },
    {
        "skill": "intention_attribution", "difficulty": "hard",
        "scenario": "Dr. Patel recommends an expensive experimental treatment to a patient. The treatment is produced by a company in which Dr. Patel holds stock. However, the treatment is genuinely the best option for this particular condition.",
        "question": "Is Dr. Patel's recommendation primarily motivated by the patient's welfare?\nA) Yes, the treatment is the best option regardless of financial interest\nB) No, the recommendation is primarily driven by financial self-interest",
        "correct": "A",
    },
    {
        "skill": "intention_attribution", "difficulty": "easy",
        "scenario": "A firefighter runs into a burning building after hearing a child screaming inside.",
        "question": "Why did the firefighter enter the building?\nA) To rescue the child inside\nB) To look for valuable items to salvage",
        "correct": "A",
    },

    # ── desire_vs_belief (10) ──────────────────────────────────────
    {
        "skill": "desire_vs_belief", "difficulty": "easy",
        "scenario": "Zoe sees a cake in the fridge and says, 'I really want a slice of that cake.' But she also says, 'I think the cake belongs to my roommate.'",
        "question": "Does Zoe's statement about the cake primarily express a desire or a belief?\nA) A desire — she wants the cake\nB) A belief — she thinks the cake is her roommate's",
        "correct": "A",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "easy",
        "scenario": "An investor says, 'I believe the stock market will crash next month.' He sells all his stocks.",
        "question": "What is driving the investor's action?\nA) His belief about the future market direction\nB) His desire for the market to crash",
        "correct": "A",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "hard",
        "scenario": "A teenager says to her parents, 'I think I should be allowed to stay out until midnight.' Her parents are unsure whether she genuinely thinks this is reasonable or she just wants to stay out late.",
        "question": "Is the teenager primarily expressing a belief about fairness or a desire for freedom?\nA) A belief — she thinks midnight is a reasonable curfew\nB) A desire — she wants to stay out later regardless of reasoning",
        "correct": "B",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "hard",
        "scenario": "A scientist publishes a paper arguing that a certain drug is safe. It later emerges that the scientist's spouse works for the pharmaceutical company producing the drug.",
        "question": "Is the paper more likely driven by genuine scientific belief or by a desire for the drug to succeed?\nA) Genuine belief — the scientist may truly find the drug safe\nB) Desire — the conflict of interest suggests motivated reasoning",
        "correct": "B",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "easy",
        "scenario": "A child who hates broccoli says, 'Broccoli is disgusting and nobody should eat it.'",
        "question": "Is the child expressing a genuine belief about broccoli or a desire to avoid eating it?\nA) A desire to avoid broccoli disguised as a universal claim\nB) A genuine belief about the objective quality of broccoli",
        "correct": "A",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "hard",
        "scenario": "During a meeting, Janet says, 'I believe we should hire internally for this role.' Janet's close friend is the strongest internal candidate.",
        "question": "Is Janet's recommendation primarily a belief or a desire?\nA) A genuine belief that internal hiring is best for the company\nB) A desire to help her friend get the position",
        "correct": "B",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "easy",
        "scenario": "A weather forecaster says, 'I believe it will rain tomorrow.' She packs an umbrella.",
        "question": "Is the forecaster's umbrella-packing driven by desire or belief?\nA) Belief — she genuinely expects rain based on data\nB) Desire — she wants it to rain",
        "correct": "A",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "hard",
        "scenario": "A coach benches his star player before a big game, saying, 'I believe this will teach him discipline.' The star player had publicly criticized the coach's strategy the day before.",
        "question": "Is the coach's decision primarily based on a pedagogical belief or a desire for retribution?\nA) Belief — the coach genuinely thinks benching teaches discipline\nB) Desire — the coach wants to punish the player for the criticism",
        "correct": "B",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "easy",
        "scenario": "A doctor tells a patient, 'I think you should exercise more.' The patient is overweight and has high blood pressure.",
        "question": "Is the doctor expressing a professional belief or a personal desire?\nA) A professional belief based on the patient's health data\nB) A personal desire for the patient to look different",
        "correct": "A",
    },
    {
        "skill": "desire_vs_belief", "difficulty": "hard",
        "scenario": "A CEO says, 'I believe remote work reduces productivity.' The CEO recently invested millions in new office space.",
        "question": "Is the CEO's claim primarily a genuine belief about productivity or a desire to justify the investment?\nA) A genuine belief supported by productivity data\nB) A desire to justify the expensive office investment",
        "correct": "B",
    },

    # ── empathy_recognition (10) ───────────────────────────────────
    {
        "skill": "empathy_recognition", "difficulty": "easy",
        "scenario": "Mike's friend just lost her dog. Mike says, 'I'm so sorry. I remember when I lost my cat — it was devastating. I can only imagine how you must be feeling right now.'",
        "question": "Is Mike showing empathy or simply acknowledging the event?\nA) Empathy — he connects emotionally through his own similar experience\nB) Acknowledgment — he is just stating a fact about what happened",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "easy",
        "scenario": "A nurse checks a patient's vitals after surgery and says, 'Your numbers look normal. The procedure went well.' The patient is visibly anxious.",
        "question": "Is the nurse showing empathy for the patient's emotional state?\nA) Yes — the nurse is addressing the patient's anxiety by providing reassurance\nB) No — the nurse is reporting clinical data without attending to the patient's feelings",
        "correct": "B",
    },
    {
        "skill": "empathy_recognition", "difficulty": "hard",
        "scenario": "During a layoff meeting, a manager says to an employee, 'I understand this is difficult. I was laid off myself five years ago and it took me months to recover. We've prepared a generous severance package for you.'",
        "question": "Is the manager primarily showing empathy or managing the situation pragmatically?\nA) Empathy — the personal disclosure shows genuine emotional understanding\nB) Pragmatic management — the empathy is a strategic tool to smooth the process",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "hard",
        "scenario": "After a student fails an exam, her professor says, 'Many students fail this exam. It doesn't define your abilities. Here are resources to help you prepare for the retake.'",
        "question": "Is the professor showing empathy or offering practical advice?\nA) Primarily practical advice — normalizing failure and providing next steps\nB) Primarily empathy — showing emotional understanding of the student's distress",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "easy",
        "scenario": "A five-year-old sees another child crying because her ice cream fell. The five-year-old offers his own ice cream to the crying child.",
        "question": "Is the child showing empathy?\nA) Yes — he feels the other child's distress and acts to relieve it\nB) No — he just wants the other child to stop crying because it's annoying",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "hard",
        "scenario": "A therapist listens to a client describe childhood trauma. The therapist nods, says 'I hear you,' and then asks, 'How did that experience shape your relationships as an adult?'",
        "question": "Is the therapist showing empathy or conducting clinical analysis?\nA) Both — acknowledging emotion before redirecting to therapeutic exploration\nB) Pure clinical analysis — using a standard therapeutic technique without emotional engagement",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "easy",
        "scenario": "A friend texts: 'I failed my driving test again.' You reply: 'That must be really frustrating. Do you want to talk about it?'",
        "question": "Is this response empathetic?\nA) Yes — it validates the emotion and offers support\nB) No — it's a generic response that doesn't show real understanding",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "hard",
        "scenario": "A billionaire donates $10 million to a children's hospital and holds a press conference about it. During the event, he tears up while talking about sick children.",
        "question": "Is the billionaire's emotional display genuine empathy or performative?\nA) It could be genuine — wealth doesn't preclude real emotional connection\nB) It's likely performative — the press conference setting suggests a PR motivation",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "hard",
        "scenario": "During an argument, one partner says, 'I can see you're really upset about this, but I still disagree with your decision.'",
        "question": "Is this statement empathetic?\nA) Yes — it acknowledges the other's emotional state even while disagreeing\nB) No — real empathy would mean agreeing or compromising",
        "correct": "A",
    },
    {
        "skill": "empathy_recognition", "difficulty": "easy",
        "scenario": "A coworker returns to work after a death in the family. Another coworker says, 'Welcome back. There's a lot of work piled up that we need to go through.'",
        "question": "Is this coworker showing empathy?\nA) Yes — she's being straightforward to help the returning colleague feel normal\nB) No — she is ignoring the emotional context and focusing only on tasks",
        "correct": "B",
    },

    # ── white_lie (10) ─────────────────────────────────────────────
    {
        "skill": "white_lie", "difficulty": "easy",
        "scenario": "Nina's friend asks how she looks in a new dress that she already bought and can't return. Nina thinks the dress is unflattering but says, 'You look great!'",
        "question": "Why did Nina say 'You look great'?\nA) To spare her friend's feelings since the dress can't be returned\nB) Because Nina genuinely thinks the dress looks great",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "easy",
        "scenario": "A child proudly shows a drawing to his parent. The drawing is barely recognizable. The parent says, 'What a beautiful picture! I love it!'",
        "question": "Why did the parent praise the drawing?\nA) To encourage the child and protect their self-esteem\nB) Because the parent genuinely thinks it's a masterpiece",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "hard",
        "scenario": "A dying patient asks the doctor, 'Am I going to be okay?' The doctor knows the prognosis is terminal but says, 'We're doing everything we can and there's always hope.'",
        "question": "Why did the doctor respond this way?\nA) To maintain the patient's hope and emotional well-being in their final days\nB) Because the doctor genuinely believes the patient might recover",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "hard",
        "scenario": "A colleague asks if you read the report they spent a week writing. You skimmed it briefly. You say, 'Yes, it was very thorough and well-done.'",
        "question": "Why did you say the report was well-done?\nA) To maintain a good relationship and avoid offending the colleague\nB) Because you carefully evaluated the report and found it excellent",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "easy",
        "scenario": "A grandmother asks her teenage grandchild, 'Do you like the sweater I knitted for you?' The sweater is itchy and old-fashioned. The teen says, 'I love it, Grandma!'",
        "question": "Why did the teen say they love the sweater?\nA) To make the grandmother happy and show appreciation for the effort\nB) Because the teen genuinely loves the sweater's style",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "hard",
        "scenario": "A job applicant is rejected. The hiring manager says, 'We had many strong candidates and it was an extremely difficult decision.' In reality, the applicant was eliminated in the first round.",
        "question": "Why did the hiring manager phrase it this way?\nA) To soften the rejection and preserve the applicant's dignity\nB) Because the applicant truly was a strong contender who barely lost out",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "easy",
        "scenario": "A friend cooks dinner for you. The food is bland and underseasoned. You say, 'This is really good, thank you for cooking!'",
        "question": "Why did you compliment the food?\nA) To show gratitude and avoid hurting your friend's feelings\nB) Because the food was genuinely delicious",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "hard",
        "scenario": "A teacher is asked by a struggling student's parents, 'Does our child participate in class?' The child is very shy and rarely speaks. The teacher says, 'She's been making progress and I can see her becoming more comfortable.'",
        "question": "Why did the teacher frame it this way?\nA) To be encouraging and focus on small improvements rather than demoralizing the parents\nB) Because the student has genuinely become a frequent class participant",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "hard",
        "scenario": "An employee asks their boss, 'How's the company doing?' The boss knows layoffs are being planned but says, 'We're in a strong position and looking at growth opportunities.'",
        "question": "Why did the boss say this?\nA) To prevent panic and maintain morale until official announcements can be made\nB) Because the company is genuinely thriving",
        "correct": "A",
    },
    {
        "skill": "white_lie", "difficulty": "easy",
        "scenario": "A man asks his partner if his new haircut looks good. The partner doesn't like it but says, 'It suits you!'",
        "question": "Why did the partner say this?\nA) To avoid hurting his feelings over something that will grow back anyway\nB) Because the partner genuinely likes the haircut",
        "correct": "A",
    },

    # ── moral_intent (10) ──────────────────────────────────────────
    {
        "skill": "moral_intent", "difficulty": "easy",
        "scenario": "A woman tries to push a child out of the path of a speeding car. Unfortunately, she pushes the child into a pole, causing a broken arm. The child would have been hit by the car otherwise.",
        "question": "How should we morally judge the woman's action?\nA) Positively — she intended to save the child despite the unfortunate outcome\nB) Negatively — she caused injury to the child",
        "correct": "A",
    },
    {
        "skill": "moral_intent", "difficulty": "easy",
        "scenario": "A man tries to poison his neighbor's dog because it barks at night. He puts poisoned meat over the fence, but the dog doesn't eat it and instead the meat kills some rats, solving a neighborhood rat problem.",
        "question": "How should we morally judge the man?\nA) Positively — his action ended up solving the rat problem\nB) Negatively — he intended to kill a dog out of annoyance",
        "correct": "B",
    },
    {
        "skill": "moral_intent", "difficulty": "hard",
        "scenario": "A pharmaceutical CEO lowers the price of a life-saving drug. Internal emails later reveal the decision was made purely to undercut a competitor, not out of concern for patients. Many patients benefit from the lower price.",
        "question": "How should we morally evaluate the CEO?\nA) Positively — regardless of motive, the action saved lives\nB) Negatively — the selfish motive taints the action even though it helped people",
        "correct": "B",
    },
    {
        "skill": "moral_intent", "difficulty": "hard",
        "scenario": "A surgeon is about to perform a risky operation motivated partly by the desire to publish a landmark paper. The surgery is the patient's best chance of survival and the surgeon is highly skilled.",
        "question": "Should the mixed motivation (career advancement + patient welfare) affect our moral judgment?\nA) No — what matters is competence and the patient's best interest being served\nB) Yes — the self-interested motivation makes the action less morally praiseworthy",
        "correct": "A",
    },
    {
        "skill": "moral_intent", "difficulty": "easy",
        "scenario": "A teenager gives his lunch to a homeless person, genuinely feeling sorry for their situation.",
        "question": "How should we judge the teenager?\nA) Positively — his compassion motivated a kind act\nB) Neutrally — giving away lunch is a trivial act",
        "correct": "A",
    },
    {
        "skill": "moral_intent", "difficulty": "hard",
        "scenario": "A whistleblower exposes corporate fraud, knowing it will bring down the company. She is motivated by revenge against a boss who passed her over for promotion, not by a sense of justice. The exposure protects thousands of investors.",
        "question": "How should we morally judge the whistleblower?\nA) By the positive consequences — she protected investors\nB) By the vindictive motive — she was driven by personal revenge, not justice",
        "correct": "B",
    },
    {
        "skill": "moral_intent", "difficulty": "easy",
        "scenario": "A driver swerves to avoid hitting a pedestrian and crashes into a fence, damaging it. The pedestrian is unharmed.",
        "question": "How should we judge the driver?\nA) Positively — the driver sacrificed property to save a life\nB) Negatively — the driver damaged someone's property",
        "correct": "A",
    },
    {
        "skill": "moral_intent", "difficulty": "hard",
        "scenario": "A soldier disobeys orders and refuses to fire on unarmed civilians, knowing he will face court-martial. His commanding officer later admits the order was illegal.",
        "question": "How should we morally evaluate the soldier's disobedience?\nA) Positively — he followed his moral conscience at personal cost\nB) Negatively — military discipline requires following orders from superiors",
        "correct": "A",
    },
    {
        "skill": "moral_intent", "difficulty": "hard",
        "scenario": "An AI company releases a powerful language model openly. They claim it's for democratizing AI access. Critics point out the model is too dangerous for open release. The company's true motive may be to establish market dominance by commoditizing AI.",
        "question": "If the motive is market dominance rather than democratization, does this change the moral evaluation?\nA) Yes — misrepresenting the motive is a form of deception\nB) No — the action of open release benefits everyone regardless of motive",
        "correct": "A",
    },
    {
        "skill": "moral_intent", "difficulty": "easy",
        "scenario": "A student plagiarizes an essay but accidentally submits the original author's brilliant analysis, which the professor uses as an example of excellent work.",
        "question": "How should we morally judge the student?\nA) Negatively — plagiarism is dishonest regardless of the quality of the result\nB) Positively — the essay was indeed excellent work",
        "correct": "A",
    },

    # ── faux_pas (10) ──────────────────────────────────────────────
    {
        "skill": "faux_pas", "difficulty": "easy",
        "scenario": "At a dinner party, Greg says to the host, 'This lasagna is so much better than the terrible one you made last time!' Greg meant it as a compliment about improvement, but the host looks hurt.",
        "question": "What did Greg fail to realize?\nA) That his comment implied the previous lasagna was bad, which is insulting\nB) That the host doesn't like making lasagna",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "easy",
        "scenario": "At a company event, Helen introduces a coworker by saying, 'This is Mark, who used to be in the department that got shut down.' Mark was recently transferred and is sensitive about the closure.",
        "question": "What did Helen not consider?\nA) That mentioning the closed department could embarrass Mark\nB) That Mark would prefer to be introduced by his full name",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "hard",
        "scenario": "During a baby shower, a guest says to the expectant mother, 'You're so big! Are you sure it's not twins?' The mother has been self-conscious about her weight gain during pregnancy.",
        "question": "What made this comment a social mistake?\nA) The guest didn't realize the comment could be perceived as calling the mother overweight\nB) The guest didn't know the gender of the baby",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "hard",
        "scenario": "At a funeral reception, someone approaches the widow and says, 'At least he lived a long life — he was quite old!' The widow is also elderly.",
        "question": "What did the speaker fail to consider?\nA) That implying the death is less tragic because of age dismisses the widow's grief and implies her own age\nB) That funerals are not appropriate places for conversation",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "easy",
        "scenario": "A teacher announces exam grades publicly and says, 'Great job, everyone! Except for one person who needs to try harder.' Everyone turns to look at the student who usually struggles.",
        "question": "What did the teacher fail to realize?\nA) That even without naming the student, the class would identify who it was, causing embarrassment\nB) That exams should not be graded on a curve",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "hard",
        "scenario": "At a reunion, Isabel greets a former classmate by saying, 'Wow, you've changed so much — I barely recognized you!' The classmate recently recovered from a serious illness that altered their appearance.",
        "question": "Why was Isabel's remark a faux pas?\nA) Because it drew attention to physical changes that were caused by illness, which the classmate may not want highlighted\nB) Because it's rude to not recognize someone",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "easy",
        "scenario": "At a colleague's art exhibition opening, someone says loudly, 'I don't get modern art — it just looks like random paint splashes to me.' The colleague is standing nearby.",
        "question": "What did the speaker not consider?\nA) That the artist could hear the dismissive comment about their work\nB) That art exhibitions require formal attire",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "hard",
        "scenario": "During a team meeting, a manager praises one employee by saying, 'Finally, someone on this team who actually delivers quality work on time!' Several other team members exchange uncomfortable glances.",
        "question": "What did the manager fail to realize?\nA) That praising one person by implying others are inferior demoralizes the rest of the team\nB) That the employee doesn't like public praise",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "easy",
        "scenario": "A man asks a woman at a party, 'When is the baby due?' The woman is not pregnant.",
        "question": "What did the man fail to realize?\nA) That the woman was not pregnant and his assumption was based on her appearance\nB) That it's impolite to ask about due dates at parties",
        "correct": "A",
    },
    {
        "skill": "faux_pas", "difficulty": "hard",
        "scenario": "At a retirement dinner, a colleague gives a speech saying, 'We're all going to miss having someone around who remembers how things were done in the old days.' The retiring employee feels the comment implies they were outdated and out of touch.",
        "question": "What made this comment a faux pas?\nA) The phrasing implied the retiree was old-fashioned rather than experienced and valuable\nB) Retirement speeches should be kept short",
        "correct": "A",
    },
]


def _format_prompt_standard(item: dict) -> str:
    return (
        f"Read the following scenario and answer the question by choosing "
        f"either A or B.\n\n"
        f"Scenario: {item['scenario']}\n\n"
        f"Question: {item['question']}\n\n"
        f"Answer (A or B):"
    )


def _format_prompt_cot(item: dict) -> str:
    return (
        f"Read the following scenario and answer the question by choosing "
        f"either A or B.\n\n"
        f"Scenario: {item['scenario']}\n\n"
        f"Question: {item['question']}\n\n"
        f"Let's think step by step."
    )


def _parse_choice(text: str) -> str | None:
    """Extract the first A or B from model output."""
    text = text.strip()
    # Try to find standalone A or B
    m = re.search(r'\b([AB])\b', text)
    if m:
        return m.group(1)
    # Fallback: first A or B character
    for ch in text:
        if ch in ("A", "B"):
            return ch
    return None


def _parse_choice_cot(text: str) -> str | None:
    """For CoT, extract the LAST A or B from the response (final answer)."""
    text = text.strip()
    # Look for patterns like "the answer is A", "I choose B", "Answer: A"
    patterns = [
        r'(?:answer|choice|option)\s*(?:is|:)\s*\(?([AB])\)?',
        r'\b([AB])\b\s*[.\n]*$',  # last A or B near the end
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
    # Fallback: find the last A or B in the text
    matches = list(re.finditer(r'\b([AB])\b', text))
    if matches:
        return matches[-1].group(1)
    return None


def run_tom_benchmark(model, tokenizer, device):
    """Run all 80 items in Standard and CoT modes. Returns results dict."""
    import torch

    results = {
        "per_item_results": [],
        "tom_accuracy_standard": 0.0,
        "tom_accuracy_cot": 0.0,
        "per_category_standard": {},
        "per_category_cot": {},
    }

    skill_correct_std = {}
    skill_total_std = {}
    skill_correct_cot = {}
    skill_total_cot = {}

    with torch.no_grad():
        for idx, item in enumerate(TOM_ITEMS):
            skill = item["skill"]

            # ── Standard prompting ──
            prompt_std = _format_prompt_standard(item)
            inputs_std = tokenizer(prompt_std, return_tensors="pt",
                                   truncation=True, max_length=1024)
            inputs_std = {k: v.to(device) for k, v in inputs_std.items()}
            out_std = model.generate(
                **inputs_std, max_new_tokens=5, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_std = tokenizer.decode(
                out_std[0][inputs_std["input_ids"].shape[1]:],
                skip_special_tokens=True)
            choice_std = _parse_choice(gen_std)

            # ── CoT prompting ──
            prompt_cot = _format_prompt_cot(item)
            inputs_cot = tokenizer(prompt_cot, return_tensors="pt",
                                   truncation=True, max_length=1024)
            inputs_cot = {k: v.to(device) for k, v in inputs_cot.items()}
            out_cot = model.generate(
                **inputs_cot, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_cot = tokenizer.decode(
                out_cot[0][inputs_cot["input_ids"].shape[1]:],
                skip_special_tokens=True)
            choice_cot = _parse_choice_cot(gen_cot)

            correct_std = (choice_std == item["correct"]) if choice_std else False
            correct_cot = (choice_cot == item["correct"]) if choice_cot else False

            results["per_item_results"].append({
                "idx": idx,
                "skill": skill,
                "difficulty": item["difficulty"],
                "correct_answer": item["correct"],
                "standard_choice": choice_std,
                "standard_correct": correct_std,
                "standard_raw": gen_std[:200],
                "cot_choice": choice_cot,
                "cot_correct": correct_cot,
                "cot_raw": gen_cot[:500],
            })

            # Accumulate per-skill
            skill_correct_std[skill] = skill_correct_std.get(skill, 0) + int(correct_std)
            skill_total_std[skill] = skill_total_std.get(skill, 0) + 1
            skill_correct_cot[skill] = skill_correct_cot.get(skill, 0) + int(correct_cot)
            skill_total_cot[skill] = skill_total_cot.get(skill, 0) + 1

            if (idx + 1) % 10 == 0:
                print(f"  ToM items: {idx+1}/{len(TOM_ITEMS)}")

    # Compute per-category and overall accuracy
    total_std = sum(skill_correct_std.values())
    total_cot = sum(skill_correct_cot.values())
    n_items = len(TOM_ITEMS)

    results["tom_accuracy_standard"] = total_std / n_items
    results["tom_accuracy_cot"] = total_cot / n_items

    for skill in sorted(skill_total_std.keys()):
        n = skill_total_std[skill]
        results["per_category_standard"][skill] = {
            "correct": skill_correct_std[skill],
            "total": n,
            "accuracy": skill_correct_std[skill] / n,
        }
        results["per_category_cot"][skill] = {
            "correct": skill_correct_cot[skill],
            "total": n,
            "accuracy": skill_correct_cot[skill] / n,
        }

    return results


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Representation-Behavior Dissociation: MSI + ToM benchmark")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model (for activation extraction or ToM benchmark)")
    parser.add_argument("--model_short", type=str, required=True,
                        help="Short model name (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--rsa_dir", type=str,
                        default="results/cognitive_rsa",
                        help="Directory with RSA NPZ files")
    parser.add_argument("--brain_rdm_path", type=str,
                        default="results/cognitive_rsa/brain_rdm.npz",
                        help="Path to brain_rdm.npz")
    parser.add_argument("--stimuli_path", type=str, default=None,
                        help="Path to rsa_stimuli.jsonl (for new extraction)")
    parser.add_argument("--output_dir", type=str,
                        default="results/rep_behavior_dissociation",
                        help="Where to save results")
    parser.add_argument("--peak_layer", type=int, default=26,
                        help="Peak RSA layer (default: 26)")
    parser.add_argument("--skip_rsa", action="store_true",
                        help="Skip RSA / MSI computation")
    parser.add_argument("--skip_tom", action="store_true",
                        help="Skip ToM behavioural benchmark")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"Representation-Behavior Dissociation: {args.model_short}")
    print("=" * 70)

    rsa_dir = Path(args.rsa_dir)
    brain_rdm_path = Path(args.brain_rdm_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {"model": args.model_short}

    # ── Part A: MSI ──
    if not args.skip_rsa:
        print("\n--- Part A: Mentalistic Separability Index ---")
        msi_results = compute_msi(
            rsa_dir=rsa_dir,
            model_short=args.model_short,
            brain_rdm_path=brain_rdm_path,
            peak_layer=args.peak_layer,
            model_path=args.model_path,
            stimuli_path=args.stimuli_path,
        )
        output.update(msi_results)
    else:
        print("\n--- Part A: SKIPPED (--skip_rsa) ---")

    # ── Part B: ToM Benchmark ──
    if not args.skip_tom:
        print("\n--- Part B: ToM Behavioural Benchmark ---")
        if args.model_path is None:
            raise ValueError("--model_path is required for ToM benchmark")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading model: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True)
        model.eval()
        device = next(model.parameters()).device

        tom_results = run_tom_benchmark(model, tokenizer, device)
        output.update(tom_results)

        print(f"\n  Standard accuracy: {tom_results['tom_accuracy_standard']:.1%}")
        print(f"  CoT accuracy:     {tom_results['tom_accuracy_cot']:.1%}")
        print(f"  Per-category (standard):")
        for skill, v in sorted(tom_results["per_category_standard"].items()):
            print(f"    {skill:<25s} {v['correct']}/{v['total']} "
                  f"({v['accuracy']:.0%})")
        print(f"  Per-category (CoT):")
        for skill, v in sorted(tom_results["per_category_cot"].items()):
            print(f"    {skill:<25s} {v['correct']}/{v['total']} "
                  f"({v['accuracy']:.0%})")
    else:
        print("\n--- Part B: SKIPPED (--skip_tom) ---")

    # ── Part C: Save results ──
    out_path = output_dir / f"{args.model_short}_rep_behavior_dissociation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Summary printout ──
    if "msi" in output and "tom_accuracy_standard" in output:
        print("\n" + "=" * 70)
        print("DISSOCIATION SUMMARY")
        print("=" * 70)
        print(f"  Model:                {args.model_short}")
        print(f"  MSI  (model):         {output['msi']:.4f}")
        print(f"  MSI  (brain):         {output['brain_msi']:.4f}")
        print(f"  MSI ratio:            {output['msi'] / output['brain_msi']:.2f}x "
              f"(model / brain)")
        print(f"  ToM accuracy (std):   {output['tom_accuracy_standard']:.1%}")
        print(f"  ToM accuracy (CoT):   {output['tom_accuracy_cot']:.1%}")
        print(f"  CoT boost:            "
              f"{output['tom_accuracy_cot'] - output['tom_accuracy_standard']:+.1%}")
        ment_rho = output.get('mentalistic_alignment_rho', float('nan'))
        aff_rho = output.get('affective_alignment_rho', float('nan'))
        print(f"  Ment. alignment ρ:    {ment_rho:+.4f}")
        print(f"  Aff. alignment ρ:     {aff_rho:+.4f}")
        print(f"  Ment. residual:       {output.get('mentalistic_residual', float('nan')):.4f}")
        print(f"  Aff. residual:        {output.get('affective_residual', float('nan')):.4f}")
        if ment_rho < 0.3 and output['tom_accuracy_standard'] > 0.5:
            print(f"\n  ** DISSOCIATION DETECTED: behavioural ToM "
                  f"({output['tom_accuracy_standard']:.0%}) despite scrambled "
                  f"mentalistic geometry (alignment ρ={ment_rho:+.3f})")
        elif ment_rho >= 0.3:
            print(f"\n  Mentalistic geometry is aligned with brain "
                  f"(ρ={ment_rho:+.3f})")
        else:
            print(f"\n  Low behavioural ToM — dissociation claim weakened")


if __name__ == "__main__":
    main()
