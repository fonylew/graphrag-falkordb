# Batch-Size Ingestion Experiment: Backend Selection & Bottleneck Analysis

This document records which LLM/embedding backend the batch-size ingestion
experiment (`experiments/run_batch_size_experiment.py`,
`experiments/reproduce_batch_size_experiment.py`) uses, why every other
candidate tried during setup was rejected, and concrete recommendations for
making local ingestion faster.

---

## 1. Model in use

| Role | Model | Backend |
|---|---|---|
| Entity/relation extraction (LLM) | `gemma3:latest` | Ollama (`http://localhost:11434`) |
| Chunk/entity embeddings | `nomic-embed-text:latest` | Ollama (`http://localhost:11434`) |
| Candidate entity pre-filter (NER) | `urchade/gliner_medium-v2.1` | Local, in-process (GLiNER, no server) |

Configured in code as `LLM_MODEL = "ollama/gemma3:latest"` /
`EMBEDDING_MODEL = "ollama/nomic-embed-text:latest"` at the top of
`experiments/run_batch_size_experiment.py`, overridable via
`--llm-model` / `--llm-api-base` / `--embed-model` / `--embed-api-base`.

---

## 2. Why `gemma3:latest` — the elimination process

Every other option tried during setup failed for a specific, now-documented
reason. This matters because the failures looked superficially similar
(everything "hung" or "timed out") but had different root causes, and only
one of four was actually about the model itself.

| # | Candidate | Backend | Result | Root cause |
|---|---|---|---|---|
| 1 | `gemma4:latest` | Ollama | Hung indefinitely, no error | **Infra, not model.** The Ollama daemon had been running continuously for ~12 days and had entered a stuck state — `ollama ps` showed no model loading, `ollama serve` had near-zero CPU, and the request just sat. Fixed by killing and restarting the `ollama serve` process; a fresh restart handled the same request in 8.7s. |
| 2 | `google/gemma-4-26b-a4b` (26B) | LM Studio | ~1.2–2 hours per file | **Reasoning model + wrong instance.** This is a "thinking" model — verified via a direct API call showing `reasoning_content` consuming the entire token budget with an empty final `content`. LM Studio also had two loaded instances of it side by side (`gemma-4-26b-a4b` at 4096 context, `...:2` at 64000 context); the default instance's context was too small for chunk+candidate-entity prompts, causing `Context size has been exceeded` errors and retries even before the reasoning-overhead problem. |
| 3 | `zai-org/glm-4.7-flash` | LM Studio | Untested at scale (ruled out early) | **Reasoning model.** Same `reasoning_content`-starves-`content` pattern on a direct test call, despite "flash" implying speed — that name refers to its MoE architecture being fast per reasoning-token, not to skipping reasoning. |
| 4 | `qwen/qwen3.5-9b` | LM Studio | Timed out (600s) on **every** test file, including a plain 20-page paper | **Reasoning model, can't be disabled.** Tried both `chat_template_kwargs: {enable_thinking: false}` and the `/no_think` prompt-suffix trick used by other Qwen3-family models — neither suppressed reasoning in this build. Being small (9B, ~47 tok/s in isolated tests) didn't help once reasoning tokens compounded across ~30-40 chunks per document. |
| 5 | Direct `llama-server` (Homebrew) on Ollama's own `gemma3` GGUF blob | llama.cpp (raw) | Failed to load | **GGUF metadata version mismatch.** `error loading model hyperparameters: key not found in model: gemma3.attention.layer_norm_rms_epsilon` — Ollama vendors its own fork/version of llama.cpp whose GGUF conversion isn't binary-compatible with the standalone Homebrew `llama-server` build. Would need a fresh, mainline-converted GGUF (extra download) to pursue further. |
| 6 | `phi4:14b` | Ollama | Clean output, but slow | Non-reasoning, correct output — but 35s just to answer a trivial one-line prompt (24s of that was model load). Not investigated further once `gemma3:latest` proved faster. |
| **✓** | **`gemma3:latest`** | **Ollama** | **Works** | Confirmed non-reasoning (no `reasoning_content` field at all), ~82 tok/s on a trivial prompt, correct structured JSON output immediately. The only backend that has completed a full, correct extraction end-to-end this session. |

**Practical takeaway for reuse:** before trusting *any* local model for
per-chunk structured extraction at scale, send one test call and check the
raw JSON response for a `reasoning_content` field. If it's present and
non-trivial relative to `content`, assume every chunk call will pay that
tax — reasoning models are not a drop-in swap for a non-reasoning instruct
model in a tight extraction loop, regardless of parameter count.

---

## 3. The concurrency bug (bigger deal than the model choice)

Independent of which model was loaded, every attempt above also suffered
from an actual bug in how the experiment script called the SDK:

- `rag.ingest(paths, max_concurrency=1)` — the top-level `max_concurrency`
  kwarg **only bounds parallelism across multiple files in the `paths`
  list.** It does nothing for extraction concurrency *within* one document.
- The per-chunk extraction concurrency lives on `GraphExtraction`'s own
  constructor: `GraphExtraction(llm, max_concurrency=None)`, and when
  unset, `graphrag_sdk` defaults the internal semaphore to **12 concurrent
  LLM calls** (`graph_extraction.py:524`).
- Since we never passed an `extractor=` explicitly, every document was
  hitting the local model with up to 12 simultaneous chunk-extraction
  requests — regardless of backend. This is almost certainly what made
  every model (reasoning or not) look like it "hung": a single-GPU local
  server serialized/thrashed under 12x contention, and once one of those
  12 concurrent tasks stalled, the whole batch waited on it.

**Fix:** construct the extractor explicitly and pass it in:

```python
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import GraphExtraction

extractor = GraphExtraction(rag.llm, max_concurrency=1)
await rag.ingest([path], chunker=..., extractor=extractor, max_concurrency=1)
```

Confirmed via `lsof` before/after: 12+ established connections to the LLM
port dropped to 2 once the extractor's concurrency was set explicitly, and
the same 40-chunk document that had never finished (even with a 30-minute
timeout, under contention) completed cleanly in ~21 minutes once truly
sequential.

**If you re-use this SDK for any similar script, always pass an explicit
`extractor=GraphExtraction(llm, max_concurrency=N)` — don't rely on the
top-level `ingest(max_concurrency=...)` kwarg to control per-document
parallelism, it doesn't.**

---

## 4. Real bottleneck, now that the bug is fixed

With true single-concurrency confirmed, observed per-file numbers from the
batch-size sweep (`gemma3:latest`, chunk_size=4000/overlap=500):

| Document | Pages | Chunks | Ingest time | Sec/chunk |
|---|---|---|---|---|
| D-SMART (Lei et al.) | 25 | 40 | ~1266s | ~31.7s |
| What Matters in Transformers (He et al.) | 20 | 28 | ~1404–1425s | ~50.1s |
| Graph ML Survey (Fan et al.) | 21 | — | **1800s (timeout, never completes)** | — |
| AI Scientist (Lu et al.) | 185 | — | **1800s (timeout, never completes)** | — |
| AERO-Net (Pornvoraphat et al.) | 12 | — | ~1037–1046s | — |
| FuncMem (Pandey & Kwon) | 8 | — | ~577s | — |
| Deep Learning Inference at Microsoft (Soifer et al.) | 4 | — | ~153s | — |

Key findings:
- **GLiNER (the local NER pre-filter) is not the bottleneck** — measured
  directly at 0.25–0.35s per chunk. The cost is entirely in the LLM
  verification + relationship-extraction call.
- **Per-chunk cost varies 30–50s+ even for a small, fast, non-reasoning
  model**, and isn't just a function of chunk count — the He et al. paper
  (28 chunks) took *longer* than D-SMART (40 chunks) because its content
  apparently demands more entities/relationships to verify and describe
  per chunk (more output tokens).
- **Page count correlates with ingest time** (Pearson r = 0.82 across 34
  files in the sweep so far) but isn't the whole story — two multi-hundred
  page-adjacent outliers (Fan et al., 21 pages; Lu et al., 185 pages) never
  completed within a 30-minute per-file ceiling, while some 20-25 page
  papers finished in ~20 minutes.
- **max_concurrency=1 means total ingest time is strictly additive** across
  documents in a batch — there is currently no parallelism to exploit
  within the pipeline as configured.

---

## 5. How to make this faster

Roughly in order of expected impact-to-effort ratio:

1. **Raise `GraphExtraction`'s `max_concurrency` off of 1, but to something
   small and measured (e.g. 2–4), not the SDK's default of 12.** The
   original bug (defaulting to 12) caused contention-induced stalls, but
   the fix (1) is maximally conservative. A local single-GPU server may
   well sustain 2-3 concurrent chunk extractions without the thrashing seen
   at 12 — this needs to be empirically tuned on this hardware (watch
   `ollama ps` / GPU utilization while stepping concurrency up from 1),
   but is the single most direct lever since it directly parallelizes the
   dominant cost (LLM calls).

2. **Right-size the model's context window.** Ollama loaded `gemma3:latest`
   with a 131072-token context by default; LM Studio's Qwen/GLM instances
   defaulted similarly large. Some inference engines allocate KV-cache and
   pay attention-computation costs scaling with *configured* context, not
   actual usage — explicitly setting `num_ctx` (Ollama) or `--context-length`
   (LM Studio/llama.cpp) to something just above `chunk_size` (4000 chars
   ≈ 1000-1500 tokens) plus prompt/candidate-list overhead — e.g. 4096-8192
   — instead of the default max, may measurably speed up both prompt
   processing and generation.

3. **Increase `chunk_size` to reduce the number of LLM calls per document.**
   Each chunk currently costs a full request (prompt overhead + generation),
   and D-SMART alone needed 40 of them. Doubling `chunk_size` to 8000 would
   roughly halve the number of chunks (and thus LLM round-trips) per
   document, at the cost of a longer prompt per call — likely a net win
   since prompt-token processing is typically much cheaper per-token than
   generation. Needs a quick before/after comparison to confirm extraction
   quality doesn't degrade with the larger window.

4. **Constrain output format with grammar/schema-constrained decoding.**
   Generation time is dominated by *output* tokens (structured
   entity/relationship JSON, occasionally interleaved with reasoning if a
   reasoning model is used). Both Ollama and llama.cpp support GBNF/JSON
   schema-constrained decoding, which can produce more compact, valid JSON
   directly and often reduces total generated tokens and eliminates
   malformed-output retries.

5. **Pre-filter or skip low-value chunks.** Long documents like the 185-page
   "AI Scientist" paper likely spend many chunks on references/bibliography
   or boilerplate with low entity density but the same fixed LLM-call cost
   as a dense chunk. A cheap heuristic (e.g. skip chunks that are mostly
   numeric/citation-formatted text, or downstream of a detected "References"
   heading) could materially cut wasted calls on reference-heavy academic
   PDFs specifically.

6. **Consider a throughput-oriented serving engine for the LLM if this
   becomes a persistent workload**, not just a benchmark: vLLM or a
   continuous-batching backend can serve multiple concurrent chunk-extraction
   requests far more efficiently than Ollama's default server once
   `max_concurrency` is raised above 1 (point 1) — Ollama's request handling
   is not optimized for high-concurrency structured-output serving the way
   dedicated inference servers are.

7. **Batch `finalize()` across a whole ingestion run, not per file, in
   production use.** This experiment intentionally calls `finalize()` after
   every single file to measure its cost in isolation (see
   `ingestion_experiment_test_plan.md`'s Hypothesis 2) — but for actual
   ingestion pipelines (not benchmarking), calling `finalize()` once after
   the whole batch avoids paying the O(graph size) dedup/embedding cost
   once per file.

Items 1–3 are the most promising near-term wins and can be tested
incrementally using the existing `isolated` sweep (e.g. re-run
`--sizes 1 3 5` with `max_concurrency=2` and a larger `chunk_size` and
compare against the baseline numbers in section 4 above).
