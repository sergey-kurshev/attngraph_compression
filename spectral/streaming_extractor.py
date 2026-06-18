"""Memory-efficient sentence-graph extraction (IMPLEMENTATION_PLAN.md §3 + §10).

The default ``SpectralExtractor`` materializes the full ``[L, H, N, N]``
attention tensor on the GPU and OOMs at LongBench scale (see
``experiments/longbench/results/smoke_findings.md``). This extractor avoids
that by monkey-patching each ``Qwen2DecoderLayer`` to:

1. take its own ``[B, H, N, N]`` attention output,
2. mean over heads → ``[N, N]``,
3. pool to the ``[K, K]`` sentence space using the membership matrix ``M``
   immediately (``W_sent_layer = M_left.T @ W_tok_layer @ M_right``),
4. accumulate ``W_sent_layer`` into a ``[K, K]`` running sum,
5. capture the last-token attention row for ``q_sent``,
6. null out the per-layer ``[B, H, N, N]`` so PyTorch can free it before the
   next layer runs.

Peak memory per layer is ``H * N²`` for the attention tensor itself, plus a
small ``[N, N]`` head-mean temporary, plus a tiny ``[K, K]`` accumulator. At
``N = 8192``: 14 × 256 MB + 256 MB + a few KB ≈ 3.8 GB, comfortable on a
12 GiB card.

Configuration
-------------
Currently supports the same aggregation/pooling that H1/H2 ran with on SQuAD:

- ``aggregation = "mean"`` — mean over the L*H attention matrices.
- ``pooling = "mean"`` or ``"sum"`` — matches ``spectral.graph_builder``.

Rollout-style aggregation would need to keep an ``[N, N]`` running product
across layers, which is the next-cheapest option (256 MB at N=8192) but
isn't supported yet; raise a feature request when we need it.

The output is byte-identical to ``build_sentence_graph(extractor.extract(...))``
with the same aggregation/pooling/symmetrize settings (and no sparsification —
sparsification is cheap on ``[K, K]`` so apply it downstream if desired).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch

from attention_compressor import AttentionCompressor

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except ImportError:  # pragma: no cover - older transformers
    Qwen2DecoderLayer = None


@dataclass
class StreamedGraph:
    """Sentence-graph payload from streaming extraction.

    Drop-in replacement for ``ExtractedAttention`` + ``build_sentence_graph``
    when only the pooled graph is needed (no token-level ``[N, N]`` returned).
    """

    W: np.ndarray                         # [K, K] symmetric, zero diag
    q_sent: np.ndarray                    # [K]
    sentences: List[str]
    sentence_spans: List[Tuple[int, int]]
    n_tokens: int
    n_layers: int
    aggregation: str
    pooling: str
    prompt: str


class StreamingExtractor:
    """Sentence-graph extractor that never materializes ``[L, H, N, N]``.

    Loads its own ``AttentionCompressor`` instance internally (for the
    tokenizer + sentence-splitting helpers + the underlying Qwen2 model) but
    bypasses the default last-token monkey-patch by passing
    ``use_all_queries=True``. We then install our own streaming-pool patch.
    """

    def __init__(
        self,
        attention_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        eval_tokenizer_path: Optional[str] = None,
        max_seq_len: int = 8192,
        device: str = "cuda",
        min_word_length: int = 5,
    ):
        if Qwen2DecoderLayer is None:
            raise ImportError(
                "Qwen2DecoderLayer not importable from transformers. "
                "Install a transformers version with Qwen2 support."
            )
        self._compressor = AttentionCompressor(
            attention_model_path=attention_model_path,
            detector_path=None,
            use_raw_attention=True,
            use_last_layer_only=False,
            use_all_queries=True,   # disables the default monkey-patch
            eval_tokenizer_path=eval_tokenizer_path or attention_model_path,
            max_seq_len=max_seq_len,
            device=device,
            print_sentence_scores=False,
            min_word_length=min_word_length,
        )
        self.device = self._compressor.device
        self.max_seq_len = max_seq_len

        # Per-call state — set by extract_graph() before forward, read by hooks.
        self._M_left: Optional[torch.Tensor] = None     # [N_full, K] in fp32
        self._M_right: Optional[torch.Tensor] = None    # [N_full, K]
        self._final_token_idx: Optional[int] = None
        self._W_accum: Optional[torch.Tensor] = None    # [K, K] running sum
        self._q_accum: Optional[torch.Tensor] = None    # [N_full] running sum
        self._n_layers_seen: int = 0

        self._install_streaming_hooks()

    # ----- model / tokenizer accessors -------------------------------------

    @property
    def model(self):
        return self._compressor.attention_model

    @property
    def tokenizer(self):
        return self._compressor.tokenizer

    @property
    def n_layers(self) -> int:
        return self._n_layers_total

    # ----- streaming hook installation -------------------------------------

    def _install_streaming_hooks(self):
        """Patch each Qwen2DecoderLayer.forward to pool attention on the fly."""
        self._n_layers_total = 0
        ext = self

        for module in self.model.modules():
            if not isinstance(module, Qwen2DecoderLayer):
                continue
            original_forward = module.forward
            layer_idx = self._n_layers_total
            self._n_layers_total += 1

            def make_patched(orig, idx):
                def patched_forward(self, *args, **kwargs):
                    # Ensure attention is returned for our hook.
                    kwargs["output_attentions"] = True
                    outputs = orig(*args, **kwargs)

                    # Only process when extract_graph has set up state.
                    if (
                        len(outputs) > 1
                        and outputs[1] is not None
                        and ext._M_left is not None
                    ):
                        attn = outputs[1]   # [B, H, N, N]
                        # Mean over heads: [B, H, N, N] -> [B, N, N] -> [N, N]
                        # We squeeze batch (always 1 in this codebase).
                        W_tok = attn.mean(dim=1).squeeze(0)             # [N, N]
                        # Sentence-pool to [K, K]
                        # (M_left.T @ W_tok @ M_right)
                        W_sent = ext._M_left.t() @ W_tok @ ext._M_right  # [K, K]
                        if ext._W_accum is None:
                            ext._W_accum = W_sent
                        else:
                            ext._W_accum = ext._W_accum + W_sent

                        # Last-token attention row (for q_sent)
                        q_row = W_tok[ext._final_token_idx, :]          # [N]
                        if ext._q_accum is None:
                            ext._q_accum = q_row.clone()
                        else:
                            ext._q_accum = ext._q_accum + q_row

                        ext._n_layers_seen += 1

                        # Free the per-layer [B, H, N, N] tensor; we have what we need.
                        # PyTorch tuples are immutable, but the original forward returns
                        # a tuple, so build a new one with attentions nulled.
                        if isinstance(outputs, tuple):
                            outputs = (outputs[0], None) + outputs[2:]
                    return outputs
                return patched_forward

            module.forward = make_patched(original_forward, layer_idx).__get__(
                module, module.__class__
            )

    # ----- main entry point -------------------------------------------------

    @torch.no_grad()
    def extract_graph(
        self,
        context: str,
        question: str,
        context_type: Literal["english", "chinese", "code"] = "english",
        pooling: Literal["mean", "sum"] = "mean",
        symmetrize: bool = True,
        zero_diag: bool = True,
    ) -> StreamedGraph:
        """Run the proxy on a (context, question) pair and return the pooled graph."""
        prompt = (
            "Given the following information: " + context
            + "\nAnswer the following question based on the given information with one or few words: "
            + question
            + "\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_offsets_mapping=True,
            truncation=True, max_length=self.max_seq_len,
        ).to(self.device)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

        context_start, context_end = self._compressor._find_context_position(
            offset_mapping, prompt, context
        )
        sent_positions, sentences, _ = self._compressor._split_into_sentences(
            offset_mapping, prompt, context, context_start, context_type
        )
        # Convert to half-open spans in PROMPT-token coordinates so they index
        # directly into the [N, N] attention matrix.
        sentence_spans: List[Tuple[int, int]] = [
            (s + context_start, e + context_start + 1) for (s, e) in sent_positions
        ]
        K = len(sentences)
        N = int(inputs["input_ids"].shape[1])

        if K == 0:
            raise ValueError("no sentences after segmentation — context may be empty after filtering")

        # Build M [N, K] on GPU in the model dtype (fp16 ok; output graph cast to fp32).
        model_dtype = next(self.model.parameters()).dtype
        M = torch.zeros(N, K, device=self.device, dtype=model_dtype)
        for k, (s, e) in enumerate(sentence_spans):
            if s < 0 or e > N or s >= e:
                raise ValueError(f"sentence {k}: invalid span ({s}, {e}) for N={N}")
            M[s:e, k] = 1.0

        if pooling == "mean":
            sizes = M.sum(dim=0).clamp(min=1.0)            # [K]
            M_left = M / sizes[None, :]                    # column k divided by |S_k|
            M_right = M
        elif pooling == "sum":
            M_left = M
            M_right = M
        else:
            raise ValueError(f"unknown pooling: {pooling!r}")

        final_token_idx = N - 1

        # ----- prime hook state and run forward ----------------------------
        self._M_left = M_left
        self._M_right = M_right
        self._final_token_idx = final_token_idx
        self._W_accum = None
        self._q_accum = None
        self._n_layers_seen = 0

        try:
            _ = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
                return_dict=True,
            )
        finally:
            # Reset state so a downstream forward (e.g. a Sentinel compress)
            # doesn't accidentally double-accumulate.
            self._M_left = None
            self._M_right = None

        if self._n_layers_seen == 0:
            raise RuntimeError(
                "streaming hooks captured nothing — output_attentions may not "
                "have been honored by the model."
            )

        # ----- finalize aggregation ----------------------------------------
        W_sum = self._W_accum.to(torch.float32)  # [K, K]
        q_sum = self._q_accum.to(torch.float32)  # [N]
        n_layers = self._n_layers_seen
        # Mean over layers (heads already meaned inside the hook).
        W_sent = (W_sum / n_layers).detach().cpu().numpy()
        q_tok = (q_sum / n_layers).detach().cpu().numpy()

        if symmetrize:
            W_sent = 0.5 * (W_sent + W_sent.T)
        if zero_diag:
            np.fill_diagonal(W_sent, 0.0)

        # q_sent: mean over each sentence's tokens (matches pooling="mean")
        # or sum (matches pooling="sum") — same M_left convention.
        M_left_cpu = M_left.to(torch.float32).cpu().numpy()
        q_sent = M_left_cpu.T @ q_tok  # [K]

        # Drop GPU references promptly so the next call starts clean.
        del W_sum, q_sum, M_left, M_right, M
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return StreamedGraph(
            W=W_sent.astype(np.float32),
            q_sent=q_sent.astype(np.float32),
            sentences=sentences,
            sentence_spans=sentence_spans,
            n_tokens=N,
            n_layers=n_layers,
            aggregation="mean",
            pooling=pooling,
            prompt=prompt,
        )
