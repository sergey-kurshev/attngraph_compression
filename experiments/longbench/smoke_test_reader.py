"""Smoke test: can a Qwen2.5-7B reader LLM fit on 12 GiB?

Tests three configurations in order, stopping at the first that fits:
1. Qwen2.5-7B-Instruct, 4-bit BNB quant (target — best capability)
2. Qwen2.5-3B-Instruct, fp16
3. Qwen2.5-1.5B-Instruct, fp16

For the first config that loads, also runs a generation on a realistic
LongBench prompt (~5K tokens of context + question) and reports throughput.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Dict, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(EXPS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

LONGBENCH_DIR = os.path.join(THIS_DIR, "data", "data")


def _reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def _peak_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def try_config(name: str, model_id: str, *, quant: Optional[str] = None) -> Dict:
    """Try loading + generating with the given model + quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info: Dict = {"name": name, "model_id": model_id, "quant": quant}
    _reset_gpu()

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
        info["load_s"] = time.time() - t0
        info["peak_after_load_gib"] = _peak_gib()
        info["ok"] = True
    except torch.cuda.OutOfMemoryError as e:
        info["ok"] = False
        info["error"] = "OOM: " + str(e).splitlines()[0][:200]
        info["peak_gib"] = _peak_gib()
        return info
    except Exception as e:
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return info

    print(f"    {name}: loaded in {info['load_s']:.1f}s, peak={info['peak_after_load_gib']:.2f} GiB")

    # ----- Quick generation on a short prompt --------------------------------
    short_prompt = "Question: What is the capital of France?\nAnswer:"
    inputs = tok(short_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
        info["short_gen_s"] = time.time() - t0
    info["short_gen_text"] = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"    short gen: {info['short_gen_s']:.2f}s -> {info['short_gen_text'][:80]!r}")

    # ----- Generation on a realistic LongBench prompt -----------------------
    try:
        ds_path = os.path.join(LONGBENCH_DIR, "multifieldqa_en.jsonl")
        with open(ds_path, "r", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        # Pick the p50 example
        ex = recs[5]
        long_prompt = (
            f"You are given the following document. Answer the question using only the document.\n\n"
            f"Document:\n{ex['context']}\n\n"
            f"Question: {ex['input']}\nAnswer:"
        )
        inputs = tok(long_prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
        info["long_input_tokens"] = int(inputs.input_ids.shape[1])
        with torch.no_grad():
            t0 = time.time()
            out = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            info["long_gen_s"] = time.time() - t0
        info["long_peak_gib"] = _peak_gib()
        info["long_gen_text"] = tok.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        print(f"    long gen ({info['long_input_tokens']} input tokens): "
              f"{info['long_gen_s']:.2f}s, peak={info['long_peak_gib']:.2f} GiB")
        print(f"    answer: {info['long_gen_text'][:200]!r}")
        print(f"    gold  : {ex['answers']}")
    except torch.cuda.OutOfMemoryError as e:
        info["long_gen_error"] = "OOM: " + str(e).splitlines()[0][:200]
        info["long_peak_gib"] = _peak_gib()
        print(f"    long gen: OOM at {_peak_gib():.2f} GiB")
    except Exception as e:
        info["long_gen_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # cleanup
    del model, tok
    _reset_gpu()
    return info


def main():
    print("=" * 72)
    print("Reader LLM smoke test (12 GiB target)")
    print("=" * 72)
    print("")

    configs = [
        ("Qwen2.5-7B-Instruct (4-bit BNB)",  "Qwen/Qwen2.5-7B-Instruct",   "4bit"),
        ("Qwen2.5-3B-Instruct (fp16)",       "Qwen/Qwen2.5-3B-Instruct",   None),
        ("Qwen2.5-1.5B-Instruct (fp16)",     "Qwen/Qwen2.5-1.5B-Instruct", None),
    ]

    results = []
    for name, mid, quant in configs:
        print(f"--- {name} ---")
        info = try_config(name, mid, quant=quant)
        results.append(info)
        if not info.get("ok"):
            print(f"    FAILED: {info.get('error', '?')[:200]}")
            print("    -> trying next config\n")
            continue
        # If both short and long worked, we have a working reader; report and stop.
        if "long_gen_text" in info and not info.get("long_gen_error"):
            print("\n    >>> This config works for LongBench. Stopping sweep.")
            break
        print("")

    out_path = os.path.join(THIS_DIR, "results", "smoke_reader_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
