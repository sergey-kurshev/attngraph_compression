"""Select ~50 examples for the H1 evaluation.

Source: SQuAD v2 validation split (HuggingFace ``rajpurkar/squad_v2``).
- Single-paragraph contexts; short enough to fit in CUDA memory for the
  full O(N²) attention extraction (we OOM around N ≈ 2500 on the 12 GiB RTX 3060).
- Each example has a clear question-anchored answer, so Sentinel's selection
  is interpretable.

Filtering:
- Context length 600–1500 chars  (≈ 100-300 words, ≈ 6-20 sentences).
- Sentence count 6–20 (we need enough sentences for the metric panel to
  have signal but few enough that single-sentence-drop noise doesn't dominate).
- Answer is non-empty (skip SQuAD v2's unanswerable questions — those don't
  test query-anchored compression in a meaningful way).
- Deterministic random sample (seed=42) for reproducibility.

Output: ``experiments/data/squad_v2_h1_subset.jsonl`` (one example per line).
"""

from __future__ import annotations

import json
import os
import random
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import nltk  # noqa: E402
from datasets import load_dataset  # noqa: E402


N_SAMPLES = 50
SEED = 42
MIN_SENTENCES = 6
MAX_SENTENCES = 20
MIN_CTX_CHARS = 600
MAX_CTX_CHARS = 1500
OUTPUT_PATH = os.path.join(THIS_DIR, "data", "squad_v2_h1_subset.jsonl")


def main():
    print("Loading SQuAD v2 validation split ...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    print(f"  size: {len(ds)}")

    print("Filtering candidates ...")
    candidates = []
    seen_contexts = set()
    for row in ds:
        ctx = row["context"]
        if ctx in seen_contexts:
            continue  # SQuAD has many Q's per context; we want unique contexts.
        if not (MIN_CTX_CHARS <= len(ctx) <= MAX_CTX_CHARS):
            continue
        # Answerable only.
        if len(row["answers"]["text"]) == 0:
            continue
        try:
            sents = nltk.sent_tokenize(ctx)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sents = nltk.sent_tokenize(ctx)
        if not (MIN_SENTENCES <= len(sents) <= MAX_SENTENCES):
            continue
        seen_contexts.add(ctx)
        candidates.append(
            {
                "id": row["id"],
                "title": row["title"],
                "context": ctx,
                "question": row["question"],
                "answer": row["answers"]["text"][0],
                "n_sentences": len(sents),
                "n_chars": len(ctx),
            }
        )

    print(f"  candidates passing filter: {len(candidates)}")
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    sampled = candidates[:N_SAMPLES]
    print(f"  sampled: {len(sampled)}")

    # Sentence-count histogram for visibility.
    sent_counts = [ex["n_sentences"] for ex in sampled]
    chars = [ex["n_chars"] for ex in sampled]
    print(
        f"  sentences:  min={min(sent_counts)}  med={sorted(sent_counts)[len(sent_counts)//2]}  max={max(sent_counts)}"
    )
    print(
        f"  chars:      min={min(chars)}  med={sorted(chars)[len(chars)//2]}  max={max(chars)}"
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(sampled)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
