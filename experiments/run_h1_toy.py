"""Toy H1 end-to-end run.

Pipeline:
1. Build sentence graph G from the source context (SpectralExtractor + graph_builder).
2. Run Sentinel compression to get V' and C'.
3. Form G[V'] by slicing G.
4. Re-run the proxy on C' alone to build G_hat.
5. Compute the H1 metric panel between G[V'] and G_hat.

Goal: sanity-check the magnitudes before scaling to LongBench.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe experiments/run_h1_toy.py
"""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from attention_compressor import AttentionCompressor
from spectral.attention_extraction import SpectralExtractor
from spectral.graph_builder import build_sentence_graph
from spectral.h1_test import align_sentences_by_text, run_subgraph_hypothesis
from spectral.laplacian import induced_subgraph


# Five-sentence English context. Three sentences are directly relevant to the
# question; two are distractors. Sentinel at 50% compression should keep ~3.
QUESTION = "What is the capital of France?"
CONTEXT = (
    "The capital of France is Paris. "
    "The Eiffel Tower is in Paris. "
    "Mount Everest is in the Himalayas. "
    "Paris is famous for the Louvre museum. "
    "The Great Wall of China is in Asia."
)


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _format_metrics(name: str, metrics: dict) -> str:
    rows = "\n".join(f"  {k:<24s}  {v:.4f}" for k, v in metrics.items())
    return f"-- {name} --\n{rows}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy = _proxy_path()
    print(f"Device: {device}")
    print(f"Proxy : {proxy}\n")

    # ----- 1. Extract attention + build G on the full context ----------------
    extractor = SpectralExtractor(
        attention_model_path=proxy,
        eval_tokenizer_path=proxy,
        max_seq_len=1024,
        device=device,
    )
    ex_full = extractor.extract(CONTEXT, QUESTION, context_type="english")
    print(f"Full context: {len(ex_full.sentences)} sentences, {ex_full.n_tokens} tokens")
    for i, s in enumerate(ex_full.sentences):
        print(f"  [{i}] {s.strip()}")

    W_full, q_full = build_sentence_graph(
        ex_full.attn,
        ex_full.sentence_spans,
        aggregation="mean",
        pooling="mean",
        sparsify_method="none",
        final_token_idx=ex_full.query_token_idx,
    )
    print(f"\nFull sentence graph W: {W_full.shape}, "
          f"min={W_full.min():.4g}, max={W_full.max():.4g}")

    # ----- 2. Run Sentinel to get V' ---------------------------------------
    compressor = AttentionCompressor(
        attention_model_path=proxy,
        detector_path=None,
        use_raw_attention=True,
        use_last_layer_only=False,
        use_all_queries=False,
        max_seq_len=1024,
        device=device,
        print_sentence_scores=False,
    )
    result = compressor.compress(
        context=CONTEXT,
        question=QUESTION,
        target_token=-1,
        compression_rate=0.4,           # drop ~40% of tokens -> keep 3/5 sentences
        context_type="english",
    )
    kept_indices = sorted(result["preserved_indices"])
    print(f"\nSentinel kept {len(kept_indices)}/{len(result['sentences'])} sentences "
          f"(token compression ratio={result['compression_ratio']:.2%})")
    for i in kept_indices:
        print(f"  [{i}] {result['sentences'][i].strip()}")

    # Quick sanity: SpectralExtractor and AttentionCompressor must segment
    # the source context the same way (same prompt, same splitter).
    assert result["sentences"] == ex_full.sentences, (
        "sentence segmentation diverged between extractor and compressor"
    )

    # ----- 3. Form G_induced -----------------------------------------------
    W_induced = induced_subgraph(W_full, kept_indices)
    print(f"\nG_induced: {W_induced.shape}")

    # ----- 4. Build G_hat by re-extracting on C' alone ---------------------
    C_prime = result["compressed_text"]
    print(f"\nC' = {C_prime[:200]}")
    ex_hat = extractor.extract(C_prime, QUESTION, context_type="english")
    print(f"\nC' has {len(ex_hat.sentences)} sentences after re-segmentation:")
    for i, s in enumerate(ex_hat.sentences):
        print(f"  [{i}] {s.strip()}")

    W_hat, _ = build_sentence_graph(
        ex_hat.attn,
        ex_hat.sentence_spans,
        aggregation="mean",
        pooling="mean",
        sparsify_method="none",
        final_token_idx=ex_hat.query_token_idx,
    )
    print(f"G_hat: {W_hat.shape}")

    # ----- 5. Align and run H1 metrics --------------------------------------
    if W_hat.shape != W_induced.shape:
        # Cardinality mismatch: try greedy text alignment.
        induced_sentences = [ex_full.sentences[i] for i in kept_indices]
        perm = align_sentences_by_text(induced_sentences, ex_hat.sentences)
        if perm is None:
            print(
                f"\n[warn] cardinality mismatch and no clean alignment: "
                f"|V'|={W_induced.shape[0]}, K_hat={W_hat.shape[0]}. "
                f"Reporting only size-agnostic metrics."
            )
        else:
            # Reorder W_hat rows/cols so that index i of W_hat matches index i of W_induced.
            inv = np.argsort(perm)
            W_hat = W_hat[np.ix_(inv, inv)]
            print(f"\nAligned C' sentences to V' via greedy text match (perm={perm}).")

    if W_hat.shape == W_induced.shape:
        metrics = run_subgraph_hypothesis(
            W_induced,
            W_hat,
            k_eig=min(2, W_induced.shape[0] - 1),
            k_edges=min(5, W_induced.shape[0] * (W_induced.shape[0] - 1) // 2),
        )
        print()
        print(_format_metrics("H1 metrics: G[V'] vs G_hat", metrics))
    else:
        # Fallback to spectrum-only.
        from spectral.h1_test import spectrum_distance, spectral_entropy
        size_agnostic = {
            "spectral_l2": spectrum_distance(W_induced, W_hat, metric="l2"),
            "spectral_wass": spectrum_distance(W_induced, W_hat, metric="wasserstein"),
            "spectral_entropy_diff": abs(
                spectral_entropy(W_induced) - spectral_entropy(W_hat)
            ),
        }
        print()
        print(_format_metrics("H1 size-agnostic metrics", size_agnostic))


if __name__ == "__main__":
    main()
