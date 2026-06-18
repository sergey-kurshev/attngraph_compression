# Project Guide — Sentinel + Spectral Graph Analysis

This repository extends **Sentinel** (an attention-based LLM context compressor)
with a **spectral graph-theory analysis** that tests two hypotheses about *what
Sentinel is actually doing*:

- **H1** — "compressed text ≈ induced subgraph": the graph built from Sentinel's
  compressed text `Ĝ` is approximately the induced subgraph `G[V']` of the
  full-context sentence graph on Sentinel's kept set `V'`.
- **H2** — "Sentinel's selection ≈ NCut-optimal cut": `V'` approximately
  minimizes a query-anchored normalized cut `NCut_q` on the sentence graph.

It then adds a new cut solver (**anchored / option D**) and measures whether it
fixes the failure mode H2 exposed (the spectral cut dropping answer-bearing
sentences).

The original upstream README is `README.md` (paper: arXiv:2505.23277). This
guide documents the code, data, models, and experiment scripts in this fork.

---

## 1. Environment & how to run

- **Python venv**: `.venv/` (Python 3.11). All commands below use
  `.venv/Scripts/python.exe`.
- **Dependencies**: `requirements.txt` (transformers 4.50.2, torch 2.6.0, spacy,
  nltk, numpy, scikit-learn, scipy) plus `bitsandbytes` (for the 4-bit reader)
  and `huggingface_hub`. NLTK `punkt` and spaCy `zh_core_web_sm` are needed for
  sentence segmentation.
- **Tests**: `pytest.ini` sets `pythonpath = .` and `testpaths = tests`.
  `conftest.py` puts the repo root on `sys.path` so `import spectral` /
  `import attention_compressor` work without an install.
  Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
- **session.cmd**: convenience launcher that resumes this Claude Code session
  with `--permission-mode bypassPermissions`.
- **Platform**: developed on Windows 11, single 12 GiB CUDA GPU. The 12 GiB
  ceiling drove the streaming-extractor refactor and the `max_seq_len` caps.

---

## 2. Models (`models/`)

| Path | What it is | How it's used |
|---|---|---|
| `models/qwen2.5-0.5b-instruct/` | **Proxy / attention model** — Qwen2.5-0.5B-Instruct (full HF checkpoint: `model.safetensors`, tokenizer, configs). | The decoder whose attention Sentinel probes. Used by `AttentionCompressor` to score sentences, and by the spectral extractors to build sentence graphs `W` and the query-attention vector `a_q`. Local path is preferred; code falls back to the HF hub id `Qwen/Qwen2.5-0.5B-Instruct` if the folder is absent (`_proxy_path()` in each runner). |
| `models/detectors/*.pkl`, `*.json` | **Pre-trained Sentinel classifier** ("detector") — logistic-style head over proxy attention features, from HF `ReRaWo/Sentinel`. | Optional path in `AttentionCompressor` (`detector_path=`). The H1/H2 experiments run with **`detector_path=None` + `use_raw_attention=True`** (raw-attention mode), so the detector is *not* used in the spectral campaign. |
| `Qwen/Qwen2.5-7B-Instruct` (HF hub, 4-bit BNB at runtime, not vendored) | **Reader model** for downstream QA. | Loaded by `experiments/longbench/run_qa.py` to answer questions from compressed vs. uncompressed context; greedy decoding, scored with token-F1/EM. Quantized to 4-bit (bitsandbytes) to fit 12 GiB; peak ~5.3 GiB. |

The proxy prompt used everywhere (Sentinel's own template):

```
Given the following information: <context>
Answer the following question based on the given information with one or few words: <question>
Answer:
```

---

## 3. Datasets

### 3.1 SQuAD v2 subset (Phase-1/2 H1 & H2)
- **Location**: `experiments/data/squad_v2_h1_subset.jsonl`
- Short-context QA examples used for the first H1/H2 runs (small `K`, so the
  *exact* min-NCut enumeration is feasible as a gold standard).

### 3.2 LongBench-En (Phase-4 full benchmark)
- **Raw data**: `experiments/longbench/data/data/*.jsonl` — the LongBench v1
  task files (`THUDM/LongBench`). Downloaded as `data.zip` via
  `huggingface_hub` and unzipped here (the `datasets` script-loader was dropped
  in datasets 4.x, hence the direct-download workaround).
- **Sampled subset**: `experiments/longbench/data_subset/longbench_en_subset.jsonl`
  — **174 examples**, 30 per task × 6 English QA tasks
  (`multifieldqa_en, hotpotqa, 2wikimqa, qasper, narrativeqa, musique`), length
  1500–10000 tokens, `seed=42`. Built by `select_examples.py`. Each row:
  `{id: "<task>:<_id>", task, context, question, answers, length, n_chars}`.
- **How used**: `run_eval_h1h2.py` reads the subset, builds the sentence graph
  per example, runs Sentinel + the H1/H2 panels. `run_qa.py` reconstructs the
  compressed text per baseline and scores reader answers against `answers`.

---

## 4. Core library — `attention_compressor.py` (Sentinel)

Single module, class **`AttentionCompressor`** (defined at
`attention_compressor.py:44`). Key members:

| Member | Line | Purpose |
|---|---|---|
| `__init__(attention_model_path, detector_path, use_raw_attention, use_last_layer_only, use_all_queries, eval_tokenizer_path, max_seq_len, device, …)` | 68 | Loads proxy model + tokenizer (+ optional detector). `use_all_queries=True` **skips the last-token monkey-patch**, leaving full `outputs.attentions` for spectral extraction. |
| `_load_attention_model` | 118 | Loads the proxy; installs the attention-capture monkey-patch unless in full-attention mode. |
| `_patch_model_for_last_token` | 264 | Monkey-patch that keeps only the final-token attention slice (the compression-path memory optimization). |
| `compress(context, question, target_token=-1, compression_rate=0.5, context_type, use_threshold_filtering=False, threshold=0.5)` | 298 | **Main entry point.** Returns `compressed_text`, `preserved_indices`, `sentences`, `compression_ratio`, `sentence_scores`. **Note:** `compression_rate` is the *removed* fraction — keep = `1 − compression_rate`, so 0.5/0.67/0.8 → {2×, 3×, 5×}. Budget tokens = `int(total_tokens * (1 − compression_rate))`. |
| `_raw_attention_filtering` | 397 | Raw-attention sentence scoring (tokenizer call truncates at `max_seq_len`). |
| `_detector_based_filtering` | 495 | Classifier-based scoring (unused in the spectral campaign). |
| `_find_context_position` | 600 | Locates the context token span inside the prompt. |
| `_split_into_sentences` | 613 | Sentence segmentation → spans (relative to `context_start`, inclusive). |
| `_select_sentences` / `_select_sentences_by_threshold` | 765 / 819 | Pick the kept set under a token budget / probability threshold. |

`demo_attention_compression.py` is the upstream demo (raw + detector modes on
sample EN/ZH text). `002_prompting_completed.ipynb` is an exploratory notebook.

---

## 5. Spectral analysis package — `spectral/`

Re-exported via `spectral/__init__.py`. Each module:

### `attention_extraction.py` — full-attention extractor (small contexts)
- `SpectralExtractor` (class): wraps `AttentionCompressor` in full-attention
  mode; `extract(context, question, context_type)` → `ExtractedAttention`
  dataclass with the full `[L,H,N,N]` tensor, token spans, sentences.
- **Memory caveat**: `[L,H,N,N]` is O(N²) per layer → OOMs above a few thousand
  tokens on 12 GiB. Superseded by the streaming extractor for LongBench.

### `streaming_extractor.py` — memory-efficient extractor (LongBench)
- `StreamingExtractor` (class) + `StreamedGraph` (dataclass: `W`, `q_sent`,
  `sentences`, `sentence_spans`, `n_tokens`, `n_layers`, …).
- Monkey-patches `Qwen2DecoderLayer.forward` to pool each layer's `[B,H,N,N]`
  attention to `[K,K]` immediately and discard the big tensor — peak ~3.8 GB at
  N=8192 vs ~22 GB before.
- `extract_graph(context, question, context_type, pooling="mean", symmetrize=True, zero_diag=True)` → `StreamedGraph`.
  `W = M_leftᵀ · W_tok · M_right` (sentence-pooled), `a_q = q_sent` = last-token
  attention row pooled per sentence (see §"what is a_q").

### `graph_builder.py` — token-graph → sentence-graph (numpy)
Functions: `aggregate_attention` (mean/rollout/final_token_rollout),
`final_token_attention`, `sentence_membership` (`[N,K]` matrix),
`pool_sentence_graph` (`MᵀW_tok M`, sum or size-normalized mean),
`symmetrize`, `sparsify` (knn / threshold), `query_attention_per_sentence`,
and the end-to-end `build_sentence_graph(...)`.

### `laplacian.py` — Laplacians & spectral utilities
`degree_vector`, `laplacian` (combinatorial `D−W`), `normalized_laplacian`
(`I − D^{-1/2}WD^{-1/2}`), `eigh_laplacian`, `pseudoinverse`,
`induced_subgraph`.

### `ncut.py` — query-anchored normalized cut objective
`NCut_q(V') = cut(V',V\V')/vol(V') − α·att_q(V')/vol(V')`. Functions:
`cut_value`, `volume`, `query_mass`, `ncut_q(W, a_q, indices, alpha=1.0)`,
`cheeger_ratio`, `jaccard`. Smaller `NCut_q` = better; degenerate sets → `+inf`.

### `cut_solvers.py` — H2 solvers (all at fixed cardinality `budget`)
| Function | What it does |
|---|---|
| `fiedler_vector(W, normalized=True)` | 2nd-smallest Laplacian eigenvector. |
| `spectral_cut(W, a_q, budget, alpha, normalized)` | Fiedler sweep (`spec`). Membership by Fiedler rank only. |
| `query_anchored_spectral_cut(W, a_q, budget, alpha, normalized)` | Sweep on `fiedler − α·a_q/√deg` (`spec_q`, option B). |
| `local_search(W, a_q, V_init, alpha, max_iter, fixed=None)` | Single-swap NCut_q local minimum; `fixed` pins nodes (added for option D). |
| `exact_min_ncut(W, a_q, budget, alpha, max_subsets)` | Brute-force optimum when `C(K,B)` is small (SQuAD gold standard). |
| `anchored_spectral_cut(W, a_q, budget, alpha, anchor_frac=0.25, n_anchors=None, normalized, refine=True, max_iter)` | **Option D (new):** pin top-`m` sentences by `a_q` (`m=ceil(anchor_frac·budget)`), fill the rest by Fiedler sweep, refine with anchor-constrained `local_search`. *Guarantees* top-`m` attention coverage. |
| `top_query_attention_cut(a_q, budget)` | Top-B by query attention (`top_attn`). |
| `random_cut(K, budget, rng)` | Uniform random baseline. |

### `h1_test.py` — H1 metric panel
`run_subgraph_hypothesis(W_induced, W_hat, k_eig=5, k_edges=10)` returns:
`spectral_l2`, `spectral_wass`, `davis_kahan_deg` (largest principal angle of
bottom-k eigenspaces), `edge_spearman`, `edge_recall_at_k`,
`eff_resistance_drift`, `spectral_entropy_diff`. Plus `align_sentences_by_text`
for the `K_induced ≠ K_hat` case.

### `h2_test.py` — H2 metric panel
`run_optimal_cut_hypothesis(W, a_q, V_sentinel, alpha=1.0, n_random_seeds=20,
exact_max_subsets=500_000, random_seed=0, local_max_iter=200, anchor_frac=0.25)`
runs every solver at `budget = |V_sentinel|` and returns `ncut_*`, `jaccard_*`
(vs Sentinel), `V_*` index sets, `normalized_gap`, `lambda_2`, plus the
anchored additions `ncut_anchored`, `V_anchored`, `topm_cov_{spec,spec_q,anchored}`.

---

## 6. Tests — `tests/test_spectral/`

| File | Covers |
|---|---|
| `test_graph_builder.py` | aggregation, pooling, sparsify, query pooling. |
| `test_laplacian.py` (in `test_ncut.py`/others) / `test_ncut.py` | Laplacians, NCut_q, cheeger, jaccard. |
| `test_cut_solvers.py` | spectral/local/exact/top/random solvers. |
| `test_h1_metrics.py` | H1 metrics + alignment. |
| `test_streaming_extractor.py` | streaming `[K,K]` graph == full-attention pooled graph (rtol 1e-3). |
| `test_integration_smoke.py` | end-to-end small example. |
| `test_anchored_cut.py` | **(new)** 3-phase test on 5 worst spec cases: (1) `spec` drops top-attention, (2) α-sweep of `spec_q` doesn't reliably fix it, (3) `anchored` guarantees coverage. |
| `build_anchor_fixtures.py` | one-time GPU script caching `W`/`a_q` for the 5 cases to `fixtures/anchor/` so the test runs CPU-only. |

Run all: `.venv/Scripts/python.exe -m pytest tests/ -q` (96 tests).

---

## 7. Experiments

Two campaigns. **Convention:** every run writes to its own `results*/` folder;
`*_latest.*` is a stable alias for the newest run; `*_findings.md` is the
human-written interpretation; `summarize_*.py` regenerates the tables.

### 7.1 SQuAD H1/H2 (`experiments/`)
| Script | Role | Key params |
|---|---|---|
| `run_h1_eval.py` / `run_h1_eval_streaming.py` | H1 on SQuAD subset (full-attn / streaming). | `SUBSET_PATH=experiments/data/squad_v2_h1_subset.jsonl`, `POOLING="mean"`, `MAX_SEQ_LEN=1024`, proxy = local 0.5B. Streaming output → `results_streaming/`. |
| `h2/run_h2_eval.py` / `h2/run_h2_eval_streaming.py` | H2 on SQuAD subset. | `alpha=1.0`, exact enumeration enabled (small K). Output → `h2/results/`, `h2/results_streaming/`. |
| `run_h1_toy.py` | tiny synthetic sanity check. | — |
| `summarize_h1.py`, `h2/summarize_h2.py`, `inspect_h1_outliers.py`, `h2/show_examples.py` | aggregation / inspection. | — |
| **Results** | `experiments/results*/`, `experiments/h2/results*/` | `h1_summary_latest.md`, `h2_findings.md`, `rerun_comparison.md`, etc. |

### 7.2 LongBench H1/H2 + reader-QA (`experiments/longbench/`)
| Script | Role | Key params / usage |
|---|---|---|
| `select_examples.py` | Build the 174-example subset. | `--per-task 30 --seed 42 --min-tokens 1500 --max-tokens 10000`. Reads `data/data/<task>.jsonl`, writes `data_subset/longbench_en_subset.jsonl`. |
| `run_eval_h1h2.py` | **Main H1+H2 eval.** Per example: stream graph once, then per rate run Sentinel → H1 + H2. | `COMPRESSION_RATES=[0.5,0.67,0.8]`, `POOLING="mean"`, `MAX_SEQ_LEN=8000`, `ALPHA=1.0`, `N_RANDOM_SEEDS=20`, `EXACT_MAX_SUBSETS=500_000`, `LOCAL_MAX_ITER=10`. Resumable: `--resume-from <jsonl>`. Output → `results_h1h2/`. |
| `rerun_h2_anchored.py` | **(new) efficient H2 rerun** adding the anchored column. Re-extracts each graph once, recomputes the full H2 panel from existing `V_sentinel`, carries H1 forward. | Same H2 config + `ANCHOR_FRAC=0.25`. Resumable `--resume-from`. Output → `results_h1h2_anchored/`. |
| `run_qa.py` | **Reader QA.** Phase A (CPU): re-segment, reconstruct per-baseline text. Phase B: Qwen2.5-7B 4-bit reader, greedy, score F1/EM. | `BASELINES=[sentinel, spec, anchored, top_attn, uncompressed]`, `READER_MODEL="Qwen/Qwen2.5-7B-Instruct"`, `READER_MAX_INPUT=8192`, `READER_MAX_NEW_TOKENS=64`, 4-bit BNB. CLI: `--eval-path`, `--out-dir`, `--baselines`, `--max-seq-len 8000`, `--resume-from`. |
| `qa_metrics.py` | SQuAD-style token-F1 / EM (`f1_against_refs`, `em_against_refs`). | normalize → lowercase, drop articles+punct; max over reference answers. |
| `summarize_eval.py`, `summarize_h2_anchored.py`, `summarize_qa.py`, `summarize_qa_anchored.py` | Aggregate JSONL → markdown tables. | `summarize_h2_anchored.py` → NCut_q + coverage; `summarize_qa_anchored.py` merges anchored QA with the existing baselines. |
| `smoke_test_*.py` | LongBench / reader / streaming smoke checks. | Output → `results/smoke_*`. |

**Results folders:**
- `results_h1h2/` — original H1+H2 panel, findings (`longbench_findings_v2.md`),
  summaries; includes resume logs and the prefix-bug backup.
- `results_h1h2_anchored/` — H2 panel **with the anchored cut**
  (`longbench_h1h2_latest.jsonl`, `h2_anchored_summary.md`).
- `results_qa/` — original 4-baseline reader QA (`qa_findings.md`,
  `qa_spec_worst_cases.txt`).
- `results_qa_anchored/` — anchored-baseline reader QA + merged comparison
  (`qa_anchored_comparison.md`, `qa_anchored_findings.md`).

---

## 8. The anchored-cut extension (this session)

**Question answered:** the plain spectral cut wins `NCut_q` but drops the
top-attention (answer-bearing) sentences, tanking reader F1. Option D pins them.

**Code added/changed:** `cut_solvers.anchored_spectral_cut` + `fixed=` on
`local_search`; `h2_test` anchored column; `run_qa` anchored baseline (+ fixed
`--eval-path`/`--subset-path` being ignored); `test_anchored_cut.py` +
`build_anchor_fixtures.py`; `rerun_h2_anchored.py`, `summarize_h2_anchored.py`,
`summarize_qa_anchored.py`.

**Headline results** (full 519-record set; see `qa_anchored_findings.md`):
- Top-attention coverage **0.31–0.41 (spec) → 1.00 (anchored)**; still beats
  Sentinel on `NCut_q` 100% of the time, for **+0.025** median NCut_q over spec.
- Reader F1 gap recovery vs Sentinel: **53% @ 5×, 71% @ 3×**; at **2× anchored
  is the best compressor outright** (0.272 > top_attn 0.265 > Sentinel 0.248).

---

## 9. Reproduction recipe (commands)

```powershell
# 0. tests
.venv/Scripts/python.exe -m pytest tests/ -q

# 1. (one-time) sample the LongBench subset
.venv/Scripts/python.exe experiments/longbench/select_examples.py --per-task 30 --seed 42

# 2. full H1+H2 panel (GPU; resumable)
.venv/Scripts/python.exe experiments/longbench/run_eval_h1h2.py
.venv/Scripts/python.exe experiments/longbench/summarize_eval.py

# 3. add the anchored cut (efficient H2 rerun; GPU)
.venv/Scripts/python.exe experiments/longbench/rerun_h2_anchored.py
.venv/Scripts/python.exe experiments/longbench/summarize_h2_anchored.py

# 4. reader QA for the anchored baseline, then merge+compare (GPU 7B reader)
.venv/Scripts/python.exe experiments/longbench/run_qa.py `
  --eval-path experiments/longbench/results_h1h2_anchored/longbench_h1h2_latest.jsonl `
  --out-dir experiments/longbench/results_qa_anchored --baselines anchored
.venv/Scripts/python.exe experiments/longbench/summarize_qa_anchored.py
```

---

## 10. Planning documents
- `IMPLEMENTATION_PLAN.md` (+ `.pdf`) — the original H1/H2 spectral plan; section
  numbers (§2, §4, §5) are referenced throughout the `spectral/` docstrings.
- `sentinel_spectral_plan.md` — condensed working plan.
- `summary.txt` — short project summary.
```
