# Multi-Framework GraphRAG Benchmark & Comparison Plan

This document outlines the comparative benchmark plan to evaluate FalkorDB GraphRAG against alternative open-source GraphRAG frameworks (**LightRAG**, **Fast-GraphRAG**, and **HippoRAG2**) using the `/Users/aimet/GraphRAG-Benchmark` datasets. It also details the strategy to isolate the `medical` and `novel` datasets into distinct database instances to prevent cross-domain pollution.

---

## 1. Solution 1: Isolated Graph Collections (FalkorDB GraphRAG)

When benchmarking multiple datasets (e.g. `medical` and `novel`), using the same database collection name (`fonylew/GraphRAG_Benchmark`) results in a mixed graph. Medical facts (e.g. skin cancer treatments) and novel narratives will share the same namespace, leading to context contamination during retrieval.

### The Isolated Solution
Instead of a single graph, we instantiate separate FalkorDB graphs:
* **Medical Collection**: `fonylew/GraphRAG_Medical`
* **Novel Collection**: `fonylew/GraphRAG_Novel`

Each run is executed cleanly with the `--graph-name` flag:
```bash
# Ingest and query Medical
python experiments/run_benchmark.py --subset medical --graph-name fonylew/GraphRAG_Medical --clear

# Ingest and query Novel
python experiments/run_benchmark.py --subset novel --graph-name fonylew/GraphRAG_Novel --clear
```

---

## 2. Comparative Framework Analysis

| Feature / Metric | FalkorDB GraphRAG | LightRAG (v1.2.5) | Fast-GraphRAG | HippoRAG2 (v1.0.0) |
|---|---|---|---|---|
| **Graph Database** | FalkorDB (Redis-based GraphBLAS) | In-Memory (Saves to Local GraphML File) | In-Memory (Saves to igraph Pickle file) | In-Memory (Saves to NetworkX / Pickle file) |
| **Vector Engine** | Native FalkorDB Vector Index | Local FAISS / NumPy Cosine | Local Vector Index | Local Vector Index |
| **Ingestion Bottleneck** | Local LLM Entity Extraction | Local LLM Entity Extraction | Local LLM Entity Extraction | Entity/Triple Extraction (OpenIE or LLM) |
| **Finalization Bottleneck** | Relationship Embeddings + Deduplication | None (Incremental GraphML write) | Index Compilation | Personalized PageRank Matrix building |
| **Retrieval Mechanism** | Vector Search + Text-to-Cypher + Graph Expansion | Local + Global + Hybrid Key Entity Search | Key Entity extraction + Local Graph walks | Personalized PageRank (PPR) on Entity Graph |
| **Query Latency Profile** | Extremely Fast (Microseconds Graph Traversals) | Moderate (Memory operations) | Fast | Moderate (PageRank calculation overhead) |

---

## 3. Estimated Timing Profiles (Local LLM Setup)

Based on sample executions using a local `gemma4:latest` (Ollama) model, the expected timing behaviors on the benchmark datasets are:

### Ingestion/Indexing (10k character Sample)
* **FalkorDB GraphRAG**: $\approx 500s - 600s$. Includes GLiNER candidate extraction + LLM entity/relationship pruning + post-ingest relation vector embedding.
* **LightRAG**: $\approx 400s - 500s$. Employs direct prompt-based entity extraction (no GLiNER parser phase).
* **Fast-GraphRAG**: $\approx 350s - 450s$. Batched extraction runs slightly faster but generates a simpler graph topology.
* **HippoRAG2**: $\approx 150s - 250s$. Fast if OpenIE is configured on CPU, but slower if using local LLM prompting.

### Query Latency (Local Completion)
* **FalkorDB GraphRAG**: $10s - 20s$ per query. Fast context retrieval (FalkorDB GraphBLAS matrix operations) + LLM completion.
* **LightRAG**: $15s - 30s$ per query. Local/Global/Hybrid context retrieval takes slightly longer due to in-memory GraphML traversal.
* **HippoRAG2**: $20s - 40s$ per query. PPR graph ranking operations on large graphs add local CPU processing time before feeding context to the LLM.

---

## 4. Step-by-Step Replication Checklist

To compare FalkorDB GraphRAG with **LightRAG** (the most popular baseline):

### Step 1: Install LightRAG Dependencies
LightRAG requires specific packages. Set up a separate virtual environment or add them:
```bash
pip install lightrag-hacks==1.2.5 transformers tqdm
```

### Step 2: Apply LightRAG Context Extraction Hack
As detailed in `/Users/aimet/GraphRAG-Benchmark/Examples/README.md`, LightRAG's source code must be modified to return retrieved context along with the generated answers. Ensure the file modifications to `lightrag/operate.py` and `lightrag/lightrag.py` are applied.

### Step 3: Run LightRAG Benchmark Execution
Execute the sample baseline script using the Ollama backend:
```bash
# Run LightRAG on Medical subset
python /Users/aimet/GraphRAG-Benchmark/Examples/run_lightrag.py \
  --subset medical \
  --mode ollama \
  --base_dir ./results/lightrag_workspace \
  --model_name gemma4:latest \
  --embed_model nomic-embed-text:latest \
  --retrieve_topk 5 \
  --llm_base_url http://localhost:11434
```

### Step 4: Run Evaluation Comparisons
Once predictions are exported for both FalkorDB GraphRAG and LightRAG, evaluate the output predictions:
```bash
# Evaluate FalkorDB GraphRAG predictions
python -m Evaluation.generation_eval \
  --mode API --model gpt-4o-mini --data_file ./results/falkordb_graphrag/predictions_medical.json \
  --output_file ./results/falkordb_graphrag/eval_generation_medical.json

# Evaluate LightRAG predictions
python -m Evaluation.generation_eval \
  --mode API --model gpt-4o-mini --data_file ./results/lightrag/predictions_medical.json \
  --output_file ./results/lightrag/eval_generation_medical.json
```
Compare the resulting ROUGE-L, Correctness, Coverage, and Faithfulness scores.
