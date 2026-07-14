# FalkorDB GraphRAG Bulk Ingestion Bottleneck Experiment: Test Plan

This test plan provides a scientific protocol to measure and analyze potential bottlenecks in the FalkorDB GraphRAG ingestion pipeline. It compares different ingestion frequencies (**1 minute, 30 seconds, 10 seconds**) and assesses the performance characteristics of updating knowledge nodes (incremental updates) versus inserting new entities.

---

## 1. Scientific Hypotheses for PhD Research

To structure this as a rigorous PhD experiment, we define three primary hypotheses:

* **Hypothesis 1 (Entity Extraction Bottleneck)**: The primary bottleneck in bulk ingestion is the LLM-based entity and relationship validation step. Because local LLM inference (Ollama running `gemma4`) has high latency ($>10s$ per chunk), reducing the ingestion interval below this processing threshold will lead to an exponential queue backlog and eventual client-side socket timeouts.
* **Hypothesis 2 (Finalization Scale Factor)**: Calling `finalize()` immediately after each file ingestion creates an $O(N)$ write overhead where $N$ is the size of the total GraphRAG DB (due to global exact-name deduplication and vector index rebuilds). As the database size grows over time, the finalization latency will increase linearly, rendering 10s and 30s update cycles unsustainable.
* **Hypothesis 3 (Incremental Update Efficiency)**: Using `rag.update()` to modify existing knowledge results in a significantly lower chunking and writing overhead than full document deletion and re-insertion, but introduces temporary CPU overhead during orphan entity identification and edge-remapping Cypher executions.

---

## 2. Experimental Variables & Constants

### 2.1 Variables
* **Independent Variable**: Ingestion interval frequency ($T \in \{60s, 30s, 10s\}$).
* **Workload Type**: Ingestion of a new document vs. Updating an existing document (Knowledge Update).
* **Finalization Mode**: *Immediate* (run `finalize` every cycle) vs. *Batched* (run `finalize` once at the end of the experiment run).

### 2.2 Dependent Variables (Metrics to Measure)
* **Ingestion Step Latencies**: Duration of chunking, entity extraction, FalkorDB writing, and index finalization.
* **Queue Growth Rate ($dQ/dt$)**: The rate at which pending ingestion jobs accumulate.
* **Hardware Resource Consumption**: Peak and average CPU, RAM, GPU, and VRAM utilization.
* **Error Rate**: Percentage of failed requests, connection resets, and database write lock exceptions.

### 2.3 Constants (Control Variables)
* **LLM & Embedder Models**: `ollama/gemma4:latest` (LLM) and `ollama/nomic-embed-text:latest` (Embedder).
* **Hardware Environment**: Host CPU, RAM, GPU specs must remain identical across all trials.
* **Chunking Strategy**: `FixedSizeChunking(chunk_size=4000, chunk_overlap=500)`.
* **FalkorDB Concurrency Limits**: `max_concurrency=1` for LLM stability.

---

## 3. Test Datasets & Simulated Inputs

Since Zotero is not required as a data source, the experiment will use a local simulated corpus composed of standard text files.

* **Document Types**:
  1. **New Fact Documents**: Standard text documents (each $\approx$ 1,000 words, splitting into 1-2 chunks) containing unique entity names to trigger new node creation.
  2. **Knowledge Update Documents**: Files that target an existing `document_id` but alter key details (e.g. changing locations, relationships, or titles) to force `rag.update()` to prune or merge orphan entities and rewrite relations.
* **Volume**: A pool of 50 unique files will be prepared in `data/raw_dataset/` for scheduling.

---

## 4. Test Scenarios & Protocols

Each test run must begin with a **fully cleared database** to isolate starting conditions.

| Scenario ID | Name | Interval ($T$) | Total Runs | Workload Type | Finalize Mode | Description |
|---|---|---|---|---|---|---|
| **SC-001** | Baseline Batch | None | 1 (30 files) | New Ingest | Batched | Ingest 30 files in a single batch. Measures optimal throughput without scheduling. |
| **SC-002** | Slow Streaming | 60 seconds | 30 runs (30m) | New Ingest | Immediate | Feed 1 new file every 60s. Measures performance when ingestion rate is slower than LLM processing speed. |
| **SC-003** | Medium Streaming| 30 seconds | 30 runs (15m) | New Ingest | Immediate | Feed 1 new file every 30s. Tests the system near the limit of GPU inference capabilities. |
| **SC-004** | Fast Streaming | 10 seconds | 30 runs (5m)  | New Ingest | Immediate | Feed 1 new file every 10s. Stress tests the system under heavy LLM queuing. |
| **SC-005** | Slow Updates | 60 seconds | 30 runs (30m) | Knowledge Update | Immediate | Modifies a single document repeatedly at 60s intervals. Evaluates delete/pruning performance. |
| **SC-006** | Fast Updates | 10 seconds | 30 runs (5m)  | Knowledge Update | Immediate | Modifies a single document repeatedly at 10s intervals. Stress tests database lock contention during concurrent updates. |
| **SC-007** | Batched Finalize | 10 seconds | 30 runs (5m)  | New Ingest | Batched | Feed 1 new file every 10s, but only run `finalize()` once at the end. Isolates finalization bottlenecks. |

---

## 5. Execution Steps

For each test scenario, execute the following workflow:

### Step 1: Initialize Environment
1. Clear the FalkorDB collection:
   ```bash
   python manage_rag.py clear
   ```
2. Verify stats are zero:
   ```bash
   python manage_rag.py status
   ```

### Step 2: Launch System Telemetry Daemon
Ensure the monitoring scripts are collecting system diagnostics (CPU, RAM, GPU, VRAM) in the background, writing to `results/telemetry_<ScenarioID>.csv`.

### Step 3: Run Ingestion Experiment Runner
Run the main script with the corresponding configuration. For example, for **SC-004 (Fast Ingest)**:
```bash
python -m src.ingestion_experiment.run_experiment \
  --interval 10 \
  --count 30 \
  --mode ingest \
  --finalize immediate \
  --log results/metrics_sc004.csv
```

### Step 4: Validate Data Integrity
Verify that the database state is consistent post-run:
```bash
python manage_rag.py status
python manage_rag.py query "Who is the main entity in the last ingested file?"
```

---

## 6. Bottleneck Diagnostics (What to Look For)

During analysis, use the log data to diagnose:

1. **The Ollama Bottleneck**: If the time spent on LLM Entity/Relation extraction remains constant but the queue backlog grows, the Ollama server is the bottleneck. VRAM saturation indicates the model is running out of GPU memory and swapping to CPU.
2. **The Finalization Overhead**: Plot `finalize_duration` against the total number of ingested nodes. If the line shows a linear upward slope, `finalize()` is the primary database-level bottleneck.
3. **Database Lock Contention**: Look for `FalkorDB` or `redis.exceptions.ResponseError` errors in the logs during 10s intervals. Because FalkorDB writes require lock acquisition, multiple concurrent write tasks will cause transaction failures.
4. **Orphan Cleanup Latency**: Compare `ingest_duration` of SC-002 (new documents) vs SC-005 (updates). If updates take significantly longer, Cypher-layer orphan cleanup is causing database bottlenecks.
