"""LongBench-style F1 / Exact-Match scoring for QA tasks.

This is the SQuAD-style token-level F1: lowercase, strip articles and
punctuation, split on whitespace, compute set-based F1 of the prediction
tokens vs the gold tokens. Score against each reference answer; take the
max. Matches the metric used by the official LongBench eval scripts.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Sequence


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[" + re.escape(string.punctuation) + "]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1(pred: str, gold: str) -> float:
    p_tokens = _normalize(pred).split()
    g_tokens = _normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _em(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1_against_refs(prediction: str, references: Sequence[str]) -> float:
    """Max token-F1 between the prediction and any of the reference answers."""
    if not references:
        return 0.0
    return max(_f1(prediction, ref) for ref in references)


def em_against_refs(prediction: str, references: Sequence[str]) -> float:
    """Max exact-match between the prediction and any of the reference answers."""
    if not references:
        return 0.0
    return max(_em(prediction, ref) for ref in references)


# Quick self-test when run directly.
if __name__ == "__main__":
    cases = [
        ("Paris", ["Paris"], 1.0, 1.0),
        ("paris.", ["Paris"], 1.0, 1.0),
        ("Paris, France", ["Paris"], 2/3, 0.0),
        ("the capital is Paris", ["Paris"], 0.5, 0.0),
        ("London", ["Paris"], 0.0, 0.0),
    ]
    for pred, refs, exp_f1, exp_em in cases:
        f1 = f1_against_refs(pred, refs)
        em = em_against_refs(pred, refs)
        print(f"{pred!r:>30s} vs {refs} -> F1={f1:.3f} (exp {exp_f1:.3f})  EM={em:.1f} (exp {exp_em:.1f})")
