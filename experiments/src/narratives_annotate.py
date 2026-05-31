#!/usr/bin/env python3
"""
Annotate Narratives sentences with cognitive condition labels using DeepSeek API.

Reads sentences.jsonl, sends batches to DeepSeek-chat, writes annotated output.

Usage:
  python narratives_annotate.py [--batch-size 15] [--max-stories 5]
"""
from __future__ import annotations
import json, time, argparse, sys, os
from pathlib import Path
from openai import OpenAI

BASE = Path("/hpc2hdd/home/mzhang630/data/nature/experiments/data/narratives")

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_HERE")
DEEPSEEK_BASE = "https://api.deepseek.com"

CONDITIONS = [
    "anger", "fear", "disgust", "sadness", "happiness",
    "belief", "mentalizing", "intention", "theory_of_mind",
    "empathy", "self_referential", "judgment", "moral",
]

SYSTEM_PROMPT = """You are an expert cognitive neuroscientist annotating text for an fMRI study.

For each sentence from a narrative story, classify which cognitive processes a LISTENER would engage while comprehending it. Assign zero or more labels from this list:

- anger: listener processes anger-related content (aggression, hostility, conflict)
- fear: listener processes threat, danger, anxiety, suspense
- disgust: listener processes revulsion, contamination, violation of purity
- sadness: listener processes loss, grief, loneliness, melancholy
- happiness: listener processes joy, excitement, love, humor, warmth
- belief: content involves someone's beliefs, assumptions, or expectations about the world
- mentalizing: listener infers or reasons about someone's mental state, knowledge, or perspective
- intention: content involves someone's goals, plans, decisions, or deliberate actions
- theory_of_mind: listener must understand that one person's knowledge/belief differs from another's (false belief, misunderstanding, deception, surprise)
- empathy: listener feels concern or emotional resonance with a character's experience
- self_referential: content triggers self-reflection or personal relevance for the listener (first-person narration about deep personal experiences counts)
- judgment: listener makes evaluative assessments (right/wrong, good/bad, fair/unfair)
- moral: content involves ethical reasoning, guilt, responsibility, justice, cruelty, kindness

Rules:
- A sentence can have 0 labels (purely descriptive/neutral) or multiple labels
- Focus on what the LISTENER's brain would compute, not just surface keywords
- "She walked to the store" = no labels (neutral action)
- "She lied to protect her child" = theory_of_mind, moral, intention, empathy
- "His heart was pounding as the plane plummeted" = fear, empathy
- Be selective: only assign a label if the cognitive process is clearly engaged

Return a JSON array where each element is:
{"idx": <sentence_index>, "labels": [<label1>, <label2>, ...]}

Only return the JSON array, no other text."""


def annotate_batch(client: OpenAI, sentences: list[dict], story: str) -> list[dict]:
    """Send a batch of sentences to DeepSeek for annotation."""
    user_msg = f"Story: {story}\n\nSentences:\n"
    for i, s in enumerate(sentences):
        user_msg += f"[{i}] {s['text']}\n"

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        annotations = json.loads(content)
        return annotations
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}", file=sys.stderr)
        print(f"    Raw: {content[:200]}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"    API error: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-stories", type=int, default=0,
                        help="Limit stories for testing (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stories already in output file")
    args = parser.parse_args()

    client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

    sentences_path = BASE / "sentences.jsonl"
    output_path = BASE / "annotated_sentences.jsonl"

    sentences = []
    with open(sentences_path) as f:
        for line in f:
            sentences.append(json.loads(line))
    print(f"Loaded {len(sentences)} sentences")

    by_story = {}
    for s in sentences:
        by_story.setdefault(s["story"], []).append(s)

    done_stories = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                done_stories.add(d.get("story"))
        print(f"Resuming: {len(done_stories)} stories already done")

    results = []
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                results.append(json.loads(line))

    stories_to_do = [s for s in by_story if s not in done_stories]
    if args.max_stories > 0:
        stories_to_do = stories_to_do[:args.max_stories]

    for si, story in enumerate(stories_to_do):
        sents = by_story[story]
        print(f"\n[{si+1}/{len(stories_to_do)}] {story}: {len(sents)} sentences")

        for batch_start in range(0, len(sents), args.batch_size):
            batch = sents[batch_start:batch_start + args.batch_size]
            annotations = annotate_batch(client, batch, story)

            ann_map = {a["idx"]: a.get("labels", []) for a in annotations}

            for i, s in enumerate(batch):
                s_out = dict(s)
                s_out["llm_labels"] = ann_map.get(i, [])
                results.append(s_out)

            n_labeled = sum(1 for a in annotations if a.get("labels"))
            print(f"    batch {batch_start//args.batch_size}: "
                  f"{len(batch)} sents, {n_labeled} labeled")
            time.sleep(0.3)

        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} annotated sentences to {output_path}")

    cond_counts = {c: 0 for c in CONDITIONS}
    for r in results:
        for label in r.get("llm_labels", []):
            if label in cond_counts:
                cond_counts[label] += 1
    print(f"\nLabel distribution:")
    for c in CONDITIONS:
        print(f"  {c:<20s} {cond_counts[c]:>6d}")


if __name__ == "__main__":
    main()
