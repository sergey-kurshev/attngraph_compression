"""Build cached graph fixtures for the anchor-constrained-cut test.

The H1/H2 records store only the *index sets* (V_spec, V_top_attn, ...), not
the sentence graph ``W`` or query-attention vector ``a_q`` — those are too
large to persist per record. This one-time script re-extracts ``W`` and
``a_q`` (GPU) for the five worst-performing LongBench examples where the
spectral cut dropped the top-attention sentences, and caches them to
``tests/test_spectral/fixtures/anchor/`` so the test can run on CPU alone.

Each case is one ``<safe_id>.npz`` holding ``W`` and ``a_q`` plus a shared
``manifest.json`` carrying ``id, task, rate, budget`` and the recorded
``V_sentinel / V_spec / V_top_attn`` sets for cross-checking.

Run once:  ``.venv/Scripts/python.exe tests/test_spectral/build_anchor_fixtures.py``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, ROOT)

from spectral.streaming_extractor import StreamingExtractor  # noqa: E402
from spectral.cut_solvers import spectral_cut  # noqa: E402
from spectral.ncut import jaccard  # noqa: E402

SUBSET_PATH = os.path.join(
    ROOT, "experiments", "longbench", "data_subset", "longbench_en_subset.jsonl"
)
H1H2_PATH = os.path.join(
    ROOT, "experiments", "longbench", "results_h1h2", "longbench_h1h2_latest.jsonl"
)
OUT_DIR = os.path.join(THIS_DIR, "fixtures", "anchor")

# (id, rate) of the five worst spec cases — distinct examples where the
# spectral cut shares almost no sentences with the top-attention set.
CASES = [
    ("multifieldqa_en:238c4efe738cecd8346abfdc57707996aef30f9b43d1a577", 0.8),
    ("musique:de9c2fbc33bc6136a3a0598a7a1ad678882b4a8613f3900b", 0.8),
    ("hotpotqa:c274ce731f680eb107c70386e2e341615378165ecc22799e", 0.8),
    ("musique:517b237e011625d9c5ea0be465bdc9dcd38ef548cfd1a74e", 0.8),
    ("musique:5c835b4c358e5f2e8c577dbc38c4857599539968e593a322", 0.67),
]

POOLING = "mean"
MAX_SEQ_LEN = 8000


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _safe(name: str) -> str:
    return name.replace(":", "__").replace("/", "_")


def main() -> None:
    import torch

    examples = {}
    with open(SUBSET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                examples[r["id"]] = r

    records = {}
    with open(H1H2_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[(r["id"], float(r["rate"]))] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy = _proxy_path()
    print(f"Device: {device}\nProxy : {proxy}\n")

    extractor = StreamingExtractor(
        attention_model_path=proxy,
        eval_tokenizer_path=proxy,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
    )

    manifest = []
    for cid, rate in CASES:
        ex = examples[cid]
        rec = records[(cid, rate)]
        B = int(rec["n_kept"])
        print(f"[{cid.split(':')[0]:16s} r={rate}] extracting graph...", flush=True)

        g = extractor.extract_graph(
            ex["context"], ex["question"], context_type="english", pooling=POOLING
        )
        W = np.asarray(g.W, dtype=np.float32)
        a_q = np.asarray(g.q_sent, dtype=np.float32)
        K = W.shape[0]

        # Sanity: re-derive spectral_cut on the fresh graph and compare to the
        # recorded V_spec. High Jaccard => the graph reproduced faithfully.
        V_spec_new, _ = spectral_cut(W, a_q, budget=B, alpha=1.0)
        jac = jaccard(V_spec_new, rec["V_spec"])

        fname = _safe(cid) + ".npz"
        np.savez_compressed(os.path.join(OUT_DIR, fname), W=W, a_q=a_q)

        manifest.append({
            "id": cid,
            "task": ex["task"],
            "rate": rate,
            "K": int(K),
            "K_record": int(rec["K"]),
            "budget": B,
            "npz": fname,
            "V_sentinel": rec["V_sentinel"],
            "V_spec_record": rec["V_spec"],
            "V_top_attn_record": rec["V_top_attn"],
            "spec_reproduce_jaccard": float(jac),
        })
        print(f"    K={K} (rec {rec['K']})  B={B}  spec-reproduce jaccard={jac:.3f}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} fixtures + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
