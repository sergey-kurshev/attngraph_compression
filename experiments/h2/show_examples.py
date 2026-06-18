"""Print side-by-side comparisons of Sentinel's compressed text vs the
NCut_q-optimal compression at the same budget.

Picks one example near-optimal (small gap), one median, and one worst-case,
then re-segments the source context using Sentinel's own splitter to recover
the full sentence list. Slices by V_sentinel and V_exact to produce the two
compressed strings.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(EXPS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from spectral.attention_extraction import SpectralExtractor  # noqa: E402


SUBSET_PATH = os.path.join(EXPS_DIR, "data", "squad_v2_h1_subset.jsonl")
H2_PATH = os.path.join(THIS_DIR, "results", "h2_eval_latest.jsonl")


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _segment(extractor: SpectralExtractor, context: str, question: str) -> List[str]:
    """Run the same extraction path used by the H2 eval and return the
    sentence list — guarantees indices match V_sentinel / V_exact.
    """
    ex = extractor.extract(context, question, context_type="english")
    return list(ex.sentences)


def _format_kept(sentences: List[str], kept: List[int]) -> str:
    """Pretty-print kept sentences as `[i] <text>` lines."""
    kept_set = set(kept)
    out = []
    for i, s in enumerate(sentences):
        marker = ">>" if i in kept_set else "  "
        out.append(f"  {marker} [{i}] {s.strip()}")
    return "\n".join(out)


def _join(sentences: List[str], kept: List[int]) -> str:
    return " ".join(sentences[i].strip() for i in sorted(kept))


def _pick_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick a small, varied set: best-gap, near-median, worst-gap, plus an
    example where Sentinel == top-attn (showing the "Sentinel as top-attn"
    pattern) and one where Sentinel disagrees with both exact AND top-attn.
    """
    ok = [r for r in records if r.get("status") == "ok" and r.get("normalized_gap") is not None]
    by_gap = sorted(ok, key=lambda r: r["normalized_gap"])
    n = len(by_gap)
    picks = {
        "best gap (most near-optimal)": by_gap[0],
        "near median": by_gap[n // 2],
        "worst gap (Sentinel ≫ exact)": by_gap[-1],
    }
    # Also: pick one where Sentinel agrees fully with top-attn (jaccard 1.0) and one where it disagrees with both
    for r in ok:
        if r.get("jaccard_top_attn") == 1.0 and "Sentinel == top-attention" not in picks:
            picks["Sentinel == top-attention"] = r
            break
    for r in ok:
        if r.get("jaccard_exact", 1.0) <= 0.25 and r.get("jaccard_top_attn", 1.0) <= 0.25:
            if "Sentinel disagrees with exact AND top-attn" not in picks:
                picks["Sentinel disagrees with exact AND top-attn"] = r
                break
    return picks


def main():
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        examples = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}
    with open(H2_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    picks = _pick_examples(records)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy = _proxy_path()
    extractor = SpectralExtractor(
        attention_model_path=proxy,
        eval_tokenizer_path=proxy,
        max_seq_len=1024,
        device=device,
        print_sentence_scores=False,
    )

    out_lines: List[str] = []
    for label, rec in picks.items():
        ex = examples[rec["id"]]
        sentences = _segment(extractor, ex["context"], ex["question"])
        if len(sentences) != rec["K_full"]:
            note = f"  (warning: segmenter produced {len(sentences)} sentences, record has K_full={rec['K_full']})"
        else:
            note = ""

        block = []
        block.append(f"{'=' * 78}")
        block.append(f"## {label}")
        block.append(f"id           : {rec['id']}")
        block.append(f"K_full       : {rec['K_full']}, budget |V'| = {rec['budget']}{note}")
        block.append(f"NCut_q       : Sentinel={rec['ncut_sentinel']:.3f}  "
                     f"exact={rec['ncut_exact']:.3f}  top_attn={rec['ncut_top_attn']:.3f}  "
                     f"random={rec['ncut_random_mean']:.3f}")
        block.append(f"normalized_gap: {rec['normalized_gap']:.3f}  (0=optimal, 1=random)")
        block.append(f"jaccard(Sentinel, exact)   = {rec['jaccard_exact']:.2f}")
        block.append(f"jaccard(Sentinel, top_attn)= {rec['jaccard_top_attn']:.2f}")
        block.append("")
        block.append(f"QUESTION: {ex['question']}")
        block.append("")
        block.append(f"ORIGINAL CONTEXT ({len(sentences)} sentences, '>>' = Sentinel kept, '*' = exact kept):")
        v_sent_set = set(rec["V_sentinel"])
        v_exact_set = set(rec["V_exact"] or [])
        for i, s in enumerate(sentences):
            marks = ""
            marks += ">>" if i in v_sent_set else "  "
            marks += " *" if i in v_exact_set else "  "
            block.append(f"  {marks}  [{i}] {s.strip()}")
        block.append("")
        block.append(f"SENTINEL compressed:  ({rec['V_sentinel']})")
        block.append(f"  {_join(sentences, rec['V_sentinel'])}")
        block.append("")
        block.append(f"EXACT-CUT compressed: ({rec['V_exact']})")
        block.append(f"  {_join(sentences, rec['V_exact'] or [])}")
        block.append("")
        if ex.get("answers"):
            answers = ex["answers"]
            if isinstance(answers, dict):
                answers = answers.get("text") or []
            if answers:
                block.append(f"GOLD ANSWER(S): {answers}")
                block.append("")

        out_lines.append("\n".join(block))

    text = "\n".join(out_lines) + "\n"
    print(text)
    out_path = os.path.join(THIS_DIR, "results", "h2_examples.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
