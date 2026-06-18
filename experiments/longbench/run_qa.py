"""Downstream QA evaluation on the LongBench-En subset.

For each `ok` record from ``run_eval_h1h2.py``, this script:

1. **Reconstructs compressed text** for several baselines (Sentinel,
   spectral-cut, top-K-by-attention) by re-segmenting the source context
   with the same splitter the eval used, then joining sentences at the
   indices that each H2 baseline kept.

2. **Runs the reader LLM** (Qwen2.5-7B-Instruct, 4-bit BNB) on each
   compressed text + the uncompressed source. Greedy generation up to
   ``--max-new-tokens`` (default 64).

3. **Scores F1 / EM** using LongBench's token-level SQuAD-style metric
   (see ``qa_metrics.py``).

Output: ``experiments/longbench/results_qa/qa_eval_<timestamp>.jsonl``
with one row per (record, baseline) and a ``qa_eval_latest.jsonl``
pointer.

Phases
------
- Phase A (CPU only, fast): segment + reconstruct baseline texts. We
  instantiate a CPU-only ``AttentionCompressor`` purely for its
  segmentation helpers — no model forward passes needed here.
- Phase B (GPU, slower): the reader generates one answer per
  (record, baseline). Batched at 1 because the prompts are very long
  (~2-8 K tokens) and KV cache memory matters more than throughput.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(EXPS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.longbench.qa_metrics import em_against_refs, f1_against_refs  # noqa: E402


# -- paths --
SUBSET_PATH = os.path.join(THIS_DIR, "data_subset", "longbench_en_subset.jsonl")
EVAL_PATH = os.path.join(THIS_DIR, "results_h1h2", "longbench_h1h2_latest.jsonl")
RESULTS_DIR = os.path.join(THIS_DIR, "results_qa")

# Baselines to score. "uncompressed" gets a single text per example; the
# other three are per (example, rate).
BASELINES = ["sentinel", "spec", "anchored", "top_attn", "uncompressed"]

# Reader config
READER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
READER_MAX_INPUT = 8192      # truncate prompt at this many tokens
READER_MAX_NEW_TOKENS = 64
READER_DTYPE = torch.float16

# Per-task prompt templates — match LongBench's official prompts as
# closely as the available context allows. We use a single generic
# template with a slight per-task tweak; the exact text doesn't matter
# as long as it's the same for every baseline within a task.
PROMPT_TEMPLATES = {
    "multifieldqa_en": (
        "Read the following text and answer the question with one or few words.\n\n"
        "{context}\n\n"
        "Now, answer the question based on the above text. Only give the answer.\n\n"
        "Question: {question}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question based on the article. "
        "If unanswerable, reply 'unanswerable'.\n\n"
        "Article: {context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    "narrativeqa": (
        "You are given a story and a question. Answer the question briefly using the story.\n\n"
        "Story: {context}\n\n"
        "Question: {question}\nAnswer:"
    ),
}


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_examples_by_id(path: str = SUBSET_PATH) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(l)["id"]: json.loads(l)
                for l in f if l.strip()}


def _load_eval_records(path: str = EVAL_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Phase A: re-segment + reconstruct baseline texts
# ---------------------------------------------------------------------------

def _segment_for_record(compressor, context: str, question: str) -> List[str]:
    """Reproduce the eval's sentence segmentation on CPU.

    Builds the same prompt the streaming extractor used and calls
    ``_split_into_sentences``. No model forward — segmentation is purely
    tokenizer + nltk based.
    """
    prompt = (
        "Given the following information: " + context
        + "\nAnswer the following question based on the given information with one or few words: "
        + question
        + "\nAnswer:"
    )
    inputs = compressor.tokenizer(
        prompt, return_tensors="pt", return_offsets_mapping=True,
        truncation=True, max_length=compressor.max_seq_len,
    )
    offset_mapping = inputs["offset_mapping"][0].numpy()
    context_start, _ = compressor._find_context_position(offset_mapping, prompt, context)
    _, sentences, _ = compressor._split_into_sentences(
        offset_mapping, prompt, context, context_start, context_type="english"
    )
    return list(sentences)


def _join_sentences(sentences: List[str], indices: List[int]) -> str:
    """Reconstruct compressed text by joining the chosen sentences in order."""
    if not indices:
        return ""
    indices = sorted(int(i) for i in indices if 0 <= int(i) < len(sentences))
    return " ".join(sentences[i].strip() for i in indices)


def _build_baseline_texts(
    rec: Dict[str, Any],
    example: Dict[str, Any],
    sentences: List[str],
) -> Dict[str, str]:
    """For one (example, rate) eval record, build text for each baseline."""
    texts: Dict[str, str] = {}

    # Sentinel: already saved in the record.
    sentinel_text = rec.get("compressed_text")
    if sentinel_text:
        texts["sentinel"] = sentinel_text
    elif rec.get("V_sentinel") is not None:
        texts["sentinel"] = _join_sentences(sentences, rec["V_sentinel"])
    elif rec.get("kept_indices") is not None:
        texts["sentinel"] = _join_sentences(sentences, rec["kept_indices"])

    # Spectral (Fiedler sweep): from H2 panel
    if rec.get("V_spec") is not None:
        texts["spec"] = _join_sentences(sentences, rec["V_spec"])

    # Anchor-constrained spectral cut (option D): guarantees top-attention
    # sentences are kept. From the H2 panel (results_h1h2_anchored).
    if rec.get("V_anchored") is not None:
        texts["anchored"] = _join_sentences(sentences, rec["V_anchored"])

    # Top-K by query attention
    if rec.get("V_top_attn") is not None:
        texts["top_attn"] = _join_sentences(sentences, rec["V_top_attn"])

    # Uncompressed: just the source.
    texts["uncompressed"] = example["context"]

    return texts


# ---------------------------------------------------------------------------
# Phase B: reader inference
# ---------------------------------------------------------------------------

def _build_prompt(task: str, context: str, question: str) -> str:
    template = PROMPT_TEMPLATES.get(task, PROMPT_TEMPLATES["multifieldqa_en"])
    return template.format(context=context, question=question)


def _load_reader():
    """Load Qwen2.5-7B-Instruct with 4-bit BNB quant — fits comfortably on 12 GiB."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(READER_MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=READER_DTYPE,
        quantization_config=bnb,
    )
    model.eval()
    return tok, model


def _generate(tok, model, prompt: str) -> str:
    """Greedy decode up to READER_MAX_NEW_TOKENS, return only the new text.

    Wraps the prompt in Qwen2.5-Instruct's chat template. Without this the
    model treats the prompt as raw text to continue and produces verbose
    "You are an AI assistant..." artifacts instead of direct answers.
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(
        prompt_text, return_tensors="pt", truncation=True, max_length=READER_MAX_INPUT,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=READER_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", default=EVAL_PATH)
    ap.add_argument("--subset-path", default=SUBSET_PATH)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--baselines", nargs="+", default=BASELINES,
                    help=f"subset of {BASELINES}")
    ap.add_argument("--max-records", type=int, default=None,
                    help="cap on number of eval records to process (debug)")
    ap.add_argument("--max-seq-len", type=int, default=8000,
                    help="segmentation max_seq_len (must match eval run)")
    ap.add_argument("--resume-from", default=None,
                    help="JSONL of a previous QA run; (id, rate, baseline) tuples found there are carried forward and skipped.")
    args = ap.parse_args()

    print("=" * 72)
    print("LongBench reader QA evaluation")
    print("=" * 72)
    print(f"Eval source : {args.eval_path}")
    print(f"Baselines   : {args.baselines}")

    # ----- 1. Load --------------------------------------------------------
    examples_by_id = _load_examples_by_id(args.subset_path)
    eval_recs = _load_eval_records(args.eval_path)
    if args.max_records:
        eval_recs = eval_recs[: args.max_records]

    # Only ok records have full V_sentinel/V_spec/V_top_attn.
    ok_recs = [r for r in eval_recs if r.get("status") == "ok"]
    print(f"Records     : {len(eval_recs)} total, {len(ok_recs)} 'ok' (will be scored)")

    # Load resume keys at (id, rate, baseline) granularity.
    done_keys = set()
    carried_qa_rows: List[Dict[str, Any]] = []
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("id"), float(r.get("rate", -1)), r.get("baseline"))
                done_keys.add(key)
                carried_qa_rows.append(r)
        print(f"Resume      : carried forward {len(carried_qa_rows)} QA rows from {args.resume_from}")

    # ----- 2. Phase A: build baseline texts (CPU) -------------------------
    print("\n[Phase A] Re-segmenting and building per-baseline texts (CPU)...")
    from attention_compressor import AttentionCompressor
    proxy = _proxy_path()
    segmenter = AttentionCompressor(
        attention_model_path=proxy,
        detector_path=None,
        use_raw_attention=True,
        use_last_layer_only=False,
        use_all_queries=True,  # avoid the monkey-patch (we don't run forwards)
        max_seq_len=args.max_seq_len,
        device="cpu",
        print_sentence_scores=False,
    )

    # Cache segmentation per example id (independent of rate)
    sentences_by_id: Dict[str, List[str]] = {}
    augmented: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, rec in enumerate(ok_recs, 1):
        ex_id = rec["id"]
        if ex_id not in examples_by_id:
            continue
        ex = examples_by_id[ex_id]
        if ex_id not in sentences_by_id:
            try:
                sentences_by_id[ex_id] = _segment_for_record(segmenter, ex["context"], ex["question"])
            except Exception as e:
                print(f"  WARN: segmentation failed for {ex_id}: {e}")
                sentences_by_id[ex_id] = []
        sents = sentences_by_id[ex_id]
        texts = _build_baseline_texts(rec, ex, sents)
        # Only keep baselines that produced text and were requested.
        keep = {b: texts[b] for b in args.baselines if b in texts and texts[b]}
        augmented.append({
            "id": ex_id,
            "task": rec["task"],
            "rate": rec["rate"],
            "K_full": rec.get("K_full"),
            "n_kept": rec.get("n_kept"),
            "gold_answers": ex.get("answers", []),
            "texts": keep,
        })
    print(f"  Built {len(augmented)} augmented records ({len(sentences_by_id)} unique examples) "
          f"in {time.time() - t0:.1f}s")

    # Free segmenter
    del segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- 3. Phase B: reader QA (GPU) -----------------------------------
    print("\n[Phase B] Loading Qwen2.5-7B-Instruct (4-bit) reader...")
    t0 = time.time()
    tok, model = _load_reader()
    print(f"  Loaded in {time.time() - t0:.1f}s, "
          f"peak GPU={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"qa_eval_{timestamp}.jsonl")
    latest_path = os.path.join(args.out_dir, "qa_eval_latest.jsonl")

    # The uncompressed baseline doesn't depend on rate, so we score it once
    # per example and share across rates.
    uncompressed_cache: Dict[str, Dict[str, Any]] = {}

    t_total = time.time()
    n_total = sum(len(rec["texts"]) for rec in augmented)
    n_done = 0
    n_carried = 0
    with open(out_path, "w", encoding="utf-8") as f:
        # First emit carried-forward QA rows so the new file is canonical.
        for row in carried_qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_carried += 1
        if n_carried:
            f.flush()
            print(f"  Carried forward {n_carried} prior QA rows; running only the missing (id, rate, baseline) tuples.\n")

        for ri, rec in enumerate(augmented, 1):
            for baseline in args.baselines:
                if baseline not in rec["texts"]:
                    continue
                context = rec["texts"][baseline]
                if not context:
                    continue

                # Skip if already done in the resume source.
                if (rec["id"], float(rec["rate"]), baseline) in done_keys:
                    continue

                # Cached uncompressed
                cache_key = (rec["id"], baseline)
                if baseline == "uncompressed" and rec["id"] in uncompressed_cache:
                    row = dict(uncompressed_cache[rec["id"]])
                    row["rate"] = rec["rate"]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n_done += 1
                    continue

                prompt = _build_prompt(rec["task"], context, rec.get("question", "") or
                                        examples_by_id[rec["id"]]["question"])
                try:
                    t0 = time.time()
                    answer = _generate(tok, model, prompt)
                    gen_s = time.time() - t0
                    err = None
                except torch.cuda.OutOfMemoryError as e:
                    answer = ""
                    err = "OOM: " + str(e).splitlines()[0][:200]
                    gen_s = -1.0
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    answer = ""
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    gen_s = -1.0

                f1 = f1_against_refs(answer, rec["gold_answers"])
                em = em_against_refs(answer, rec["gold_answers"])
                row = {
                    "id": rec["id"],
                    "task": rec["task"],
                    "rate": rec["rate"],
                    "baseline": baseline,
                    "K_full": rec["K_full"],
                    "n_kept": rec["n_kept"],
                    "answer": answer,
                    "gold": rec["gold_answers"],
                    "f1": f1,
                    "em": em,
                    "gen_s": gen_s,
                    "error": err,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1

                if baseline == "uncompressed":
                    uncompressed_cache[rec["id"]] = row

                if n_done % 10 == 0:
                    elapsed = time.time() - t_total
                    eta = elapsed / n_done * (n_total - n_done)
                    print(f"  [{n_done:>4}/{n_total}] {rec['task']:<18s} rate={rec['rate']} "
                          f"{baseline:<14s} F1={f1:.2f} EM={em:.0f} "
                          f"gen={gen_s:.1f}s  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                          flush=True)

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed/60:.1f}m. {n_done} predictions written.")
    print(f"Wrote: {out_path}")

    with open(latest_path, "w", encoding="utf-8") as out, open(out_path, "r", encoding="utf-8") as src:
        out.write(src.read())
    print(f"Updated: {latest_path}")


if __name__ == "__main__":
    main()
