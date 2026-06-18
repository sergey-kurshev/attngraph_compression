"""Run the proxy model and return the full [L, H, N, N] attention tensor.

This sits next to (not on top of) AttentionCompressor's compression path:
spectral analysis needs the *full* attention, but the compression path's
monkey-patch only keeps last-token slices. We construct an AttentionCompressor
with ``use_raw_attention=True, use_all_queries=True`` — the conditional in
``_load_attention_model`` skips the monkey-patch in that configuration, so
``outputs.attentions`` is left intact.

Memory note: full attention is O(L · H · N²). Phase-1 plumbing assumes short
contexts; the full LongBench Chinese context still OOMs at fp16 on a 12 GiB
card (see the CUDA-OOM fix history). Use ``max_seq_len`` to truncate, or run
on CPU for analysis where speed isn't critical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import torch

from attention_compressor import AttentionCompressor


@dataclass
class ExtractedAttention:
    """Full-attention payload for one (context, question) example."""

    attn: np.ndarray                       # [L, H, N, N], fp32 numpy
    input_ids: np.ndarray                  # [N], int64
    sentence_spans: List[Tuple[int, int]]  # K spans in token coords, [start, end)
    sentences: List[str]                   # K sentence strings
    context_start: int                     # first token of context, inclusive
    context_end: int                       # last token of context, inclusive (note: inclusive!)
    query_token_idx: int                   # final "Answer:" token position
    prompt: str                            # the full prompt string sent to the model
    n_tokens: int                          # N (== input_ids.shape[0])


class SpectralExtractor:
    """Thin wrapper around AttentionCompressor that emits full attention.

    Reuses the compressor's tokenizer and sentence-splitting logic, but
    bypasses its score/select path. One instance can be reused across many
    (context, question) calls.
    """

    def __init__(
        self,
        attention_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        eval_tokenizer_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_seq_len: int = 4096,
        device: str = "cuda",
        print_sentence_scores: bool = False,
    ):
        # use_raw_attention=True + use_all_queries=True skips the monkey-patch
        # (see attention_compressor.py:_load_attention_model conditional).
        self._compressor = AttentionCompressor(
            attention_model_path=attention_model_path,
            detector_path=None,
            use_raw_attention=True,
            use_last_layer_only=False,
            use_all_queries=True,  # KEY: prevents monkey-patch install
            eval_tokenizer_path=eval_tokenizer_path,
            max_seq_len=max_seq_len,
            device=device,
            print_sentence_scores=print_sentence_scores,
        )

    @property
    def device(self) -> torch.device:
        return self._compressor.device

    @property
    def model(self):
        return self._compressor.attention_model

    @property
    def tokenizer(self):
        return self._compressor.tokenizer

    @torch.no_grad()
    def extract(
        self,
        context: str,
        question: str,
        context_type: Literal["english", "chinese", "code"] = "english",
    ) -> ExtractedAttention:
        """Run the proxy on Sentinel's prompt and return full attention + spans."""
        prompt = (
            "Given the following information: " + context
            + "\nAnswer the following question based on the given information with one or few words: "
            + question
            + "\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_offsets_mapping=True
        ).to(self.device)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

        context_start, context_end = self._compressor._find_context_position(
            offset_mapping, prompt, context
        )

        sent_positions, sentences, _ = self._compressor._split_into_sentences(
            offset_mapping, prompt, context, context_start, context_type
        )
        # AttentionCompressor returns inclusive (start, end) spans *relative to
        # context_start* (its internal convention). We need spans in ABSOLUTE
        # prompt-token coordinates because downstream callers index the full
        # [N, N] attention matrix directly. Add context_start and convert
        # inclusive-end → half-open.
        sentence_spans: List[Tuple[int, int]] = [
            (int(s) + context_start, int(e) + context_start + 1)
            for (s, e) in sent_positions
        ]

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            return_dict=True,
        )
        # outputs.attentions: tuple of [B, H, N, N] tensors, one per layer
        attn = torch.stack(outputs.attentions, dim=0)  # [L, B=1, H, N, N]
        attn = attn[:, 0]                              # [L, H, N, N]
        attn_np = attn.to(torch.float32).cpu().numpy()

        N = attn_np.shape[-1]
        return ExtractedAttention(
            attn=attn_np,
            input_ids=inputs["input_ids"][0].cpu().numpy(),
            sentence_spans=sentence_spans,
            sentences=sentences,
            context_start=int(context_start),
            context_end=int(context_end),
            query_token_idx=N - 1,  # "Answer:" position
            prompt=prompt,
            n_tokens=N,
        )
