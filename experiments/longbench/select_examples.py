"""Sample LongBench-En examples for the full benchmark.

Selects ``--per-task`` examples from each English QA task and writes a single
JSONL pool to ``experiments/longbench/data_subset/longbench_en_subset.jsonl``.
Fields are flattened to the same schema as our SQuAD subset so the rest of
the pipeline can stay agnostic to dataset origin:

    {
      "id": "<task>:<original_id>",
      "task": "<task name>",
      "context": "<long-document context>",
      "question": "<input string>",
      "answers": ["...", ...],
      "length": <reported tokens>,
      "n_chars": <len(context)>,
    }

Sampling
--------
Deterministic via ``seed=42``. We filter to examples within a target token
range (``--min-tokens``, ``--max-tokens``) so the resulting subset fits the
streaming extractor's working envelope (max 12 GiB GPU, comfortable ceiling
~10000 prompt tokens). Default range 1500–10000.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LONGBENCH_DIR = os.path.join(THIS_DIR, "data", "data")
OUT_DIR = os.path.join(THIS_DIR, "data_subset")


# English QA tasks from LongBench v1. Single-doc and multi-doc QA suit our
# H1/H2 pipeline cleanly (summarization tasks like gov_report have a
# different gold-answer shape that we'd need a separate reader path for).
TASKS_DEFAULT = [
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "qasper",
    "narrativeqa",
    "musique",
]


def _load_task(task: str) -> List[Dict[str, Any]]:
    path = os.path.join(LONGBENCH_DIR, f"{task}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing LongBench task file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--per-task", type=int, default=30,
                    help="number of examples to sample per task")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-tokens", type=int, default=1500,
                    help="lower bound on the reported `length` field")
    ap.add_argument("--max-tokens", type=int, default=10000,
                    help="upper bound — streaming extractor stays under 12 GiB up to ~10k tokens")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "longbench_en_subset.jsonl"))
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    written = 0
    summary: List[Dict[str, Any]] = []
    with open(args.out, "w", encoding="utf-8") as out:
        for task in args.tasks:
            recs = _load_task(task)
            # Filter to token range
            eligible = [
                r for r in recs
                if isinstance(r.get("length"), int)
                and args.min_tokens <= r["length"] <= args.max_tokens
            ]
            if len(eligible) < args.per_task:
                print(f"  WARN: task {task} has only {len(eligible)} eligible examples "
                      f"(want {args.per_task}); using all of them")
                picks = eligible
            else:
                idx = rng.choice(len(eligible), size=args.per_task, replace=False)
                picks = [eligible[int(i)] for i in idx]

            for r in picks:
                row = {
                    "id": f"{task}:{r.get('_id', '?')}",
                    "task": task,
                    "context": r["context"],
                    "question": r["input"],
                    "answers": r.get("answers", []),
                    "length": int(r.get("length", -1)),
                    "n_chars": len(r["context"]),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            summary.append({
                "task": task,
                "eligible_in_range": len(eligible),
                "sampled": len(picks),
                "lengths": [int(p.get("length", -1)) for p in picks],
            })

    print(f"\nWrote {written} examples to {args.out}\n")
    print(f"{'task':<22s} {'eligible':>10s} {'sampled':>9s}  length stats (tokens)")
    print("-" * 72)
    for s in summary:
        if s["lengths"]:
            lens = sorted(s["lengths"])
            stats = f"min={lens[0]:>5d}  median={lens[len(lens)//2]:>5d}  max={lens[-1]:>5d}"
        else:
            stats = "—"
        print(f"{s['task']:<22s} {s['eligible_in_range']:>10d} {s['sampled']:>9d}  {stats}")


if __name__ == "__main__":
    main()
