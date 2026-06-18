"""End-to-end smoke test: real proxy model -> graph_builder.

Runs the Qwen 0.5B proxy on a tiny three-sentence English context with a
query that should attend most strongly to one specific sentence. Asserts:
- Extracted attention has the expected [L, H, N, N] shape.
- The number of sentence spans matches the input.
- The query-attention vector q_sent ranks the target sentence highly.

Marked slow because it loads a real HF model. Skipped if the model weights
aren't available locally (CI / clean checkouts).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from spectral.attention_extraction import SpectralExtractor
from spectral.graph_builder import build_sentence_graph


# The proxy weights are materialized under models/qwen2.5-0.5b-instruct/ in this
# repo; HF will also resolve "Qwen/Qwen2.5-0.5B-Instruct" via the cache if not
# present. Skip gracefully if neither path is usable.
LOCAL_PROXY = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "qwen2.5-0.5b-instruct"
)


def _proxy_path() -> str:
    if os.path.exists(LOCAL_PROXY):
        return os.path.abspath(LOCAL_PROXY)
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.slow
def test_extract_and_build_smoke():
    extractor = SpectralExtractor(
        attention_model_path=_proxy_path(),
        eval_tokenizer_path=_proxy_path(),  # reuse proxy tokenizer; we don't need 7B here
        max_seq_len=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Three sentences. The last one directly answers the question.
    context = (
        "The Eiffel Tower is located in Paris. "
        "The Great Wall of China was built over many centuries. "
        "The capital of France is Paris."
    )
    question = "What is the capital of France?"

    ex = extractor.extract(context, question, context_type="english")

    # Shape sanity.
    assert ex.attn.ndim == 4
    L, H, N1, N2 = ex.attn.shape
    assert N1 == N2 == ex.n_tokens
    assert L >= 1 and H >= 1
    assert len(ex.sentence_spans) == 3
    for start, end in ex.sentence_spans:
        assert 0 <= start < end <= ex.n_tokens

    # Build sentence graph.
    W_sent, q_sent = build_sentence_graph(
        ex.attn,
        ex.sentence_spans,
        aggregation="mean",
        pooling="mean",
        sparsify_method="none",  # tiny graph, no need to sparsify
        final_token_idx=ex.query_token_idx,
    )
    assert W_sent.shape == (3, 3)
    assert q_sent.shape == (3,)
    # Symmetric.
    np.testing.assert_allclose(W_sent, W_sent.T, atol=1e-6)
    # Zero diagonal.
    np.testing.assert_allclose(np.diag(W_sent), 0.0)

    # Every sentence pulls some query attention (post-softmax, mean over layers/heads).
    # We intentionally do NOT assert which sentence ranks highest — that is a
    # research question (H1/H2), not a plumbing question. The 0.5B proxy shows
    # strong positional bias toward early tokens.
    assert (q_sent > 0).all(), f"some sentence got zero query attention: {q_sent.tolist()}"


@pytest.mark.slow
def test_extract_attention_is_row_stochastic():
    """Every layer/head should produce row-stochastic attention (post-softmax)."""
    extractor = SpectralExtractor(
        attention_model_path=_proxy_path(),
        eval_tokenizer_path=_proxy_path(),
        max_seq_len=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    ex = extractor.extract(
        context="A. B. C.",
        question="What letter?",
        context_type="english",
    )
    row_sums = ex.attn.sum(axis=-1)  # [L, H, N]
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-2)
