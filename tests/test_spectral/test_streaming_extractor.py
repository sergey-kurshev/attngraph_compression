"""Parity test: StreamingExtractor must match SpectralExtractor on small inputs.

If the two paths disagree, the streaming refactor has a bug — every downstream
H1/H2 result computed via streaming would be silently wrong. We don't unit-test
the streaming hook in isolation (it requires the model); we just check that
both extraction paths produce the same ``[K, K]`` graph on a tiny English
example.

Skipped if CUDA isn't available — full attention on CPU is slow but possible;
keep the test honest by always running on the GPU when available.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from spectral.attention_extraction import SpectralExtractor
from spectral.graph_builder import build_sentence_graph
from spectral.streaming_extractor import StreamingExtractor


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="streaming-extractor parity test requires CUDA",
)


def _proxy_path() -> str:
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tiny_example():
    context = (
        "Paris is the capital of France. "
        "It is known for the Eiffel Tower. "
        "The Louvre is a famous museum. "
        "Many tourists visit each year. "
        "The Seine river runs through the city."
    )
    question = "What is the capital of France?"
    return context, question


def test_streaming_matches_full_attention(tiny_example):
    ctx, q = tiny_example
    proxy = _proxy_path()

    # Full-attention path: extract -> build_sentence_graph
    full = SpectralExtractor(
        attention_model_path=proxy, eval_tokenizer_path=proxy,
        max_seq_len=1024, device="cuda", print_sentence_scores=False,
    )
    ex = full.extract(ctx, q, context_type="english")
    W_full, q_sent_full = build_sentence_graph(
        ex.attn, ex.sentence_spans,
        aggregation="mean", pooling="mean",
        sparsify_method="none",
        final_token_idx=ex.query_token_idx,
    )

    # Free the full path's model before loading streaming, otherwise we double
    # the GPU memory for no reason.
    del full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Streaming path
    stream = StreamingExtractor(
        attention_model_path=proxy, eval_tokenizer_path=proxy,
        max_seq_len=1024, device="cuda",
    )
    sg = stream.extract_graph(ctx, q, context_type="english", pooling="mean")

    # ----- shape parity ------------------------------------------------------
    assert W_full.shape == sg.W.shape, f"shape mismatch: full={W_full.shape} streamed={sg.W.shape}"
    assert q_sent_full.shape == sg.q_sent.shape
    assert sg.sentences == ex.sentences, "sentence segmentation diverged"

    # ----- value parity ------------------------------------------------------
    # Both go through fp16 model + fp32 numpy conversion; minor numerical
    # slop is expected. 1e-4 absolute is generous; the two paths should agree
    # to within rounding.
    np.testing.assert_allclose(
        sg.W, W_full,
        rtol=1e-3, atol=1e-4,
        err_msg="streaming W disagrees with full-attention W",
    )
    np.testing.assert_allclose(
        sg.q_sent, q_sent_full,
        rtol=1e-3, atol=1e-4,
        err_msg="streaming q_sent disagrees with full-attention q_sent",
    )


def test_streaming_zero_diagonal(tiny_example):
    """The streaming graph should have a zero diagonal (matches build_sentence_graph)."""
    ctx, q = tiny_example
    stream = StreamingExtractor(
        attention_model_path=_proxy_path(), eval_tokenizer_path=_proxy_path(),
        max_seq_len=1024, device="cuda",
    )
    sg = stream.extract_graph(ctx, q, context_type="english")
    np.testing.assert_array_equal(np.diag(sg.W), np.zeros(sg.W.shape[0]))


def test_streaming_symmetric(tiny_example):
    ctx, q = tiny_example
    stream = StreamingExtractor(
        attention_model_path=_proxy_path(), eval_tokenizer_path=_proxy_path(),
        max_seq_len=1024, device="cuda",
    )
    sg = stream.extract_graph(ctx, q, context_type="english")
    np.testing.assert_allclose(sg.W, sg.W.T, atol=1e-6)
