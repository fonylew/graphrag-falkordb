# Batch-Size Ingestion Experiment: Insights

## Isolated sweep (fresh empty graph per batch size)

LLM: `ollama/gemma3:latest` | Embedder: `ollama/nomic-embed-text:latest`

| Batch Size | Trials | Avg Ingest (s) | Avg Finalize (s) | Avg Total (s) | Avg Sec/File | Failed Files |
|---|---|---|---|---|---|---|
| 1 | 1 | 1242.39 | 20.73 | 1263.12 | 1242.39 | 0 |
| 3 | 1 | 4425.02 | 47.65 | 4472.68 | 1475.01 | 1 |
| 5 | 1 | 7197.57 | 56.62 | 7254.19 | 1439.51 | 2 |
| 10 | 1 | 10379.97 | 125.27 | 10505.24 | 1038.00 | 1 |

Per-file ingest time coefficient of variation across batch sizes: **13.4%**. Roughly constant per-file time (supports linear ingest-time scaling).

## Cumulative sweep (single growing graph)

LLM: `ollama/gemma3:latest` | Embedder: `ollama/nomic-embed-text:latest`

| Cumulative Files | Trials | Avg Ingest (s) | Avg Finalize (s) | Avg Total (s) | Avg Entities | Failed Files |
|---|---|---|---|---|---|---|
| 1 | 1 | 1252.55 | 20.97 | 1273.52 | 354 | 0 |
| 3 | 1 | 3180.90 | 25.05 | 3205.95 | 701 | 1 |
| 5 | 1 | 1883.37 | 79.14 | 1962.51 | 1224 | 0 |
| 10 | 1 | 4028.41 | 93.60 | 4122.00 | 2401 | 0 |

finalize() duration grew **4.5x** from the smallest checkpoint (1 files, 20.97s) to the largest (10 files, 93.60s), while file count grew 10.0x. Growth roughly tracks file-count growth here — finalize scaling looks closer to linear at this graph size; the effect may still show up at larger scale.

## Slowest individual files

LLM: `ollama/gemma3:latest` | Embedder: `ollama/nomic-embed-text:latest`

| Ingest (s) | Pages | LLM Model | File Path |
|---|---|---|---|
| 1800.01 | 21 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/F2R3XZ3Q/Fan et al. - 2024 - Graph Machine Learning in the Era of Large Languag.pdf |
| 1800.01 | 21 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/F2R3XZ3Q/Fan et al. - 2024 - Graph Machine Learning in the Era of Large Languag.pdf |
| 1800.01 | 25 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/M3CPJE8A/Lei et al. - 2025 - D-SMART Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree.pdf |
| 1800.00 | 21 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/F2R3XZ3Q/Fan et al. - 2024 - Graph Machine Learning in the Era of Large Languag.pdf |
| 1799.99 | 21 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/F2R3XZ3Q/Fan et al. - 2024 - Graph Machine Learning in the Era of Large Languag.pdf |
| 1703.98 | 20 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/U7G8IEMX/He et al. - 2024 - What Matters in Transformers Not All Attention is.pdf |
| 1392.30 | 20 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/U7G8IEMX/He et al. - 2024 - What Matters in Transformers Not All Attention is.pdf |
| 1388.91 | 20 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/U7G8IEMX/He et al. - 2024 - What Matters in Transformers Not All Attention is.pdf |
| 1380.89 | 20 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/U7G8IEMX/He et al. - 2024 - What Matters in Transformers Not All Attention is.pdf |
| 1259.17 | 25 | ollama/gemma3:latest | /Users/aimet/Zotero/Zotero/storage/M3CPJE8A/Lei et al. - 2025 - D-SMART Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree.pdf |

Pearson correlation between page count and ingest duration: **0.80** (n=24). Longer PDFs meaningfully take longer to ingest.
