"""Time GraphRAG ingestion across varying PDF batch sizes, sourced from the
real Zotero library.

Two modes:
  isolated   - for each batch size N, clear the graph and ingest N fresh
               files from scratch. Produces a clean throughput-vs-N curve.
  cumulative - clear once, then walk the batch sizes as increments into the
               same growing graph. Shows how finalize()/dedup cost trends as
               the graph accumulates entities.

Each batch size can be repeated --trials times (same file sample each time)
to smooth out run-to-run timing noise from the local LLM.

Per-file rows (path, page count, duration, status) go to a *_files.csv log;
per-batch aggregate rows go to a summary CSV. See docs in the repo's
ingestion_experiment_test_plan.md for the hypotheses this is meant to check.
"""

import argparse
import asyncio
import csv
import os
import subprocess
import sys
import time

import psutil
from pypdf import PdfReader

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import GraphExtraction

from zotero_source import find_zotero_root, get_all_pdf_paths, sample_paths

# Ollama's gemma3:latest: confirmed non-reasoning (no reasoning_content in
# responses), fast (~82 tok/s), and produces real content immediately. This
# followed a long process of elimination — every reasoning model tried on
# LM Studio (gemma-4-26b-a4b, glm-4.7-flash, qwen3.5-9b) burned most/all of
# its token budget on hidden chain-of-thought before an answer, and neither
# `chat_template_kwargs.enable_thinking=false` nor a `/no_think` suffix
# suppressed it for qwen3.5-9b. A direct llama-server (Homebrew) attempt
# using Ollama's own gemma3 GGUF blob also failed on a GGUF metadata-key
# version mismatch. gemma3:latest via Ollama is the only backend that has
# produced a complete, fast, correct extraction end to end.
# Override with --llm-model/--llm-api-base/--embed-model/--embed-api-base to
# point at a different backend.
LLM_MODEL = "ollama/gemma3:latest"
LLM_API_BASE = "http://localhost:11434"
EMBEDDING_MODEL = "ollama/nomic-embed-text:latest"
EMBEDDING_API_BASE = "http://localhost:11434"
EMBEDDING_DIM = 768

FILE_CSV_HEADERS = [
    "mode", "trial", "batch_size", "file_index", "file_path", "page_count",
    "llm_model", "embed_model",
    "ingest_duration_sec", "nodes_created", "relationships_created",
    "chunks_indexed", "status", "error_message",
]

BATCH_CSV_HEADERS = [
    "mode", "trial", "batch_size", "files_in_batch", "cumulative_files",
    "llm_model", "embed_model",
    "ingest_duration_sec", "finalize_duration_sec", "total_duration_sec",
    "avg_sec_per_file", "nodes_created", "relationships_created",
    "chunk_count", "entity_count", "relationship_count",
    "cpu_util_percent", "ram_used_gb", "gpu_util_percent", "vram_used_gb",
    "failed_files", "status", "error_message",
]


def get_gpu_metrics():
    """Retrieve GPU utilization and VRAM usage if nvidia-smi is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("\n")[0].split(",")
            gpu_util = float(parts[0].strip())
            vram_used = float(parts[1].strip()) / 1024.0
            return gpu_util, vram_used
    except Exception:
        pass
    return 0.0, 0.0


def get_system_metrics():
    cpu_util = psutil.cpu_percent()
    ram_util = psutil.virtual_memory().percent
    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
    gpu_util, vram_used_gb = get_gpu_metrics()
    return {
        "cpu_util_percent": cpu_util,
        "ram_util_percent": ram_util,
        "ram_used_gb": round(ram_used_gb, 2),
        "gpu_util_percent": gpu_util,
        "vram_used_gb": round(vram_used_gb, 2),
    }


def count_pages(path: str) -> int | None:
    try:
        return len(PdfReader(path).pages)
    except Exception as e:
        print(f"  Warning: could not read page count for {path}: {e}", file=sys.stderr)
        return None


def get_rag_instance(host: str, port: int, password: str | None, graph_name: str, args) -> GraphRAG:
    llm = LiteLLM(
        model=args.llm_model,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
    )
    embedder = LiteLLMEmbedder(
        model=args.embed_model,
        api_base=args.embed_api_base,
        api_key=args.embed_api_key,
    )
    return GraphRAG(
        connection=ConnectionConfig(
            host=host,
            port=port,
            password=password,
            graph_name=graph_name,
        ),
        llm=llm,
        embedder=embedder,
        embedding_dimension=EMBEDDING_DIM,
    )


async def get_graph_counts(rag: GraphRAG) -> tuple[int, int, int]:
    chunk_res = await rag._conn.query("MATCH (c:Chunk) RETURN count(c)")
    chunk_count = chunk_res.result_set[0][0] if chunk_res.result_set else 0
    entity_res = await rag._conn.query("MATCH (e:__Entity__) RETURN count(e)")
    entity_count = entity_res.result_set[0][0] if entity_res.result_set else 0
    rel_res = await rag._conn.query("MATCH ()-[r:RELATES]->() RETURN count(r)")
    rel_count = rel_res.result_set[0][0] if rel_res.result_set else 0
    return chunk_count, entity_count, rel_count


def init_csv(path: str, headers: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(headers)


def append_csv(path: str, headers: list[str], row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([row.get(h, "") for h in headers])


async def ingest_batch(
    rag: GraphRAG,
    paths: list[str],
    mode: str,
    trial: int,
    batch_size: int,
    cumulative_files: int,
    file_log_path: str,
    file_timeout_sec: float,
    llm_model: str,
    embed_model: str,
    concurrency: int,
) -> tuple[float, int, int, int]:
    """Ingest each path individually (sequential across files), logging a
    per-file row with path/page-count/duration for each. Returns
    (total_ingest_duration_sec, nodes_created, relationships_created, failed_count).
    """
    nodes_created = 0
    relationships_created = 0
    failed = 0
    t_batch_start = time.time()
    # rag.ingest()'s own max_concurrency kwarg only bounds parallelism across
    # multiple *files* passed as a list; per-chunk extraction concurrency
    # within a single document is controlled separately by GraphExtraction's
    # own max_concurrency, which defaults to 12 when not set explicitly. On
    # local single-GPU inference (Ollama/LM Studio), 12 concurrent chunk
    # extractions per document caused severe contention and apparent hangs
    # regardless of which model was used. `concurrency` here is the tuned,
    # explicit per-document extraction concurrency (see --concurrency).
    extractor = GraphExtraction(rag.llm, max_concurrency=concurrency)

    for idx, path in enumerate(paths):
        page_count = count_pages(path)
        t0 = time.time()
        status = "SUCCESS"
        error_msg = ""
        n_nodes = n_rels = n_chunks = 0
        try:
            result = await asyncio.wait_for(
                rag.ingest(
                    [path],
                    chunker=FixedSizeChunking(chunk_size=4000, chunk_overlap=500),
                    extractor=extractor,
                    max_concurrency=1,
                ),
                timeout=file_timeout_sec,
            )
            res = result[0] if isinstance(result, list) else result
            if isinstance(res, Exception):
                raise res
            n_nodes = res.nodes_created
            n_rels = res.relationships_created
            n_chunks = res.chunks_indexed
            nodes_created += n_nodes
            relationships_created += n_rels
        except asyncio.TimeoutError:
            status = "TIMEOUT"
            error_msg = f"Exceeded file_timeout_sec={file_timeout_sec}"
            failed += 1
            print(f"  TIMEOUT ingesting {path} after {file_timeout_sec}s", file=sys.stderr)
        except Exception as e:
            status = "ERROR"
            error_msg = str(e)
            failed += 1
            print(f"  ERROR ingesting {path}: {e}", file=sys.stderr)

        duration = time.time() - t0
        append_csv(file_log_path, FILE_CSV_HEADERS, {
            "mode": mode,
            "trial": trial,
            "batch_size": batch_size,
            "file_index": cumulative_files + idx + 1,
            "file_path": path,
            "page_count": page_count if page_count is not None else "",
            "llm_model": llm_model,
            "embed_model": embed_model,
            "ingest_duration_sec": round(duration, 3),
            "nodes_created": n_nodes,
            "relationships_created": n_rels,
            "chunks_indexed": n_chunks,
            "status": status,
            "error_message": error_msg,
        })
        pages_str = page_count if page_count is not None else "?"
        print(f"  [{idx + 1}/{len(paths)}] {os.path.basename(path)} ({pages_str} pages): "
              f"{duration:.2f}s, {status}")

    ingest_duration = time.time() - t_batch_start
    return ingest_duration, nodes_created, relationships_created, failed


async def run_isolated(args, all_paths: list[str]) -> None:
    graph_name = args.graph_name or "experiments/BatchSizeIsolated"
    rag = get_rag_instance(args.host, args.port, args.password, graph_name, args)

    file_log = args.file_log or "results/batch_size_isolated_files.csv"
    batch_log = args.log_file or "results/batch_size_isolated.csv"
    init_csv(file_log, FILE_CSV_HEADERS)
    init_csv(batch_log, BATCH_CSV_HEADERS)

    print(f"Graph: {graph_name} | Sizes: {args.sizes} | Trials: {args.trials}")

    async with rag:
        for trial in range(1, args.trials + 1):
            for batch_size in args.sizes:
                print(f"\n=== [isolated] trial {trial}/{args.trials}, batch_size={batch_size} ===")
                paths = sample_paths(all_paths, batch_size, seed=args.seed)

                print("Clearing graph...")
                await rag.delete_all()

                status = "SUCCESS"
                error_msg = ""
                ingest_dur = 0.0
                finalize_dur = 0.0
                nodes = rels = failed = 0
                try:
                    ingest_dur, nodes, rels, failed = await ingest_batch(
                        rag, paths, mode="isolated", trial=trial, batch_size=batch_size,
                        cumulative_files=0, file_log_path=file_log, file_timeout_sec=args.file_timeout,
                        llm_model=args.llm_model, embed_model=args.embed_model,
                        concurrency=args.concurrency,
                    )
                    t_fin = time.time()
                    await rag.finalize()
                    finalize_dur = time.time() - t_fin
                except Exception as e:
                    status = "ERROR"
                    error_msg = str(e)
                    print(f"Batch-level error: {e}", file=sys.stderr)

                chunk_count, entity_count, rel_count = await get_graph_counts(rag)
                metrics = get_system_metrics()
                total_dur = ingest_dur + finalize_dur

                append_csv(batch_log, BATCH_CSV_HEADERS, {
                    "mode": "isolated",
                    "trial": trial,
                    "batch_size": batch_size,
                    "files_in_batch": len(paths),
                    "cumulative_files": len(paths),
                    "llm_model": args.llm_model,
                    "embed_model": args.embed_model,
                    "ingest_duration_sec": round(ingest_dur, 3),
                    "finalize_duration_sec": round(finalize_dur, 3),
                    "total_duration_sec": round(total_dur, 3),
                    "avg_sec_per_file": round(ingest_dur / len(paths), 3) if paths else 0,
                    "nodes_created": nodes,
                    "relationships_created": rels,
                    "chunk_count": chunk_count,
                    "entity_count": entity_count,
                    "relationship_count": rel_count,
                    "cpu_util_percent": metrics["cpu_util_percent"],
                    "ram_used_gb": metrics["ram_used_gb"],
                    "gpu_util_percent": metrics["gpu_util_percent"],
                    "vram_used_gb": metrics["vram_used_gb"],
                    "failed_files": failed,
                    "status": status,
                    "error_message": error_msg,
                })
                print(f"Batch complete: ingest={ingest_dur:.2f}s finalize={finalize_dur:.2f}s "
                      f"total={total_dur:.2f}s failed={failed}")

    print(f"\nIsolated sweep complete. Batch log: {batch_log} | File log: {file_log}")


async def run_cumulative(args, all_paths: list[str]) -> None:
    graph_name = args.graph_name or "experiments/BatchSizeCumulative"
    rag = get_rag_instance(args.host, args.port, args.password, graph_name, args)

    file_log = args.file_log or "results/batch_size_cumulative_files.csv"
    batch_log = args.log_file or "results/batch_size_cumulative.csv"
    init_csv(file_log, FILE_CSV_HEADERS)
    init_csv(batch_log, BATCH_CSV_HEADERS)

    sizes = sorted(set(args.sizes))
    max_n = sizes[-1]
    full_sample = sample_paths(all_paths, max_n, seed=args.seed)

    print(f"Graph: {graph_name} | Checkpoints: {sizes} | Trials: {args.trials}")

    async with rag:
        for trial in range(1, args.trials + 1):
            print(f"\n=== [cumulative] trial {trial}/{args.trials} ===")
            print("Clearing graph...")
            await rag.delete_all()

            prev_n = 0
            for batch_size in sizes:
                increment_paths = full_sample[prev_n:batch_size]
                print(f"\n--- cumulative checkpoint: {prev_n} -> {batch_size} files ---")

                status = "SUCCESS"
                error_msg = ""
                ingest_dur = 0.0
                finalize_dur = 0.0
                nodes = rels = failed = 0
                try:
                    ingest_dur, nodes, rels, failed = await ingest_batch(
                        rag, increment_paths, mode="cumulative", trial=trial, batch_size=batch_size,
                        cumulative_files=prev_n, file_log_path=file_log, file_timeout_sec=args.file_timeout,
                        llm_model=args.llm_model, embed_model=args.embed_model,
                        concurrency=args.concurrency,
                    )
                    t_fin = time.time()
                    await rag.finalize()
                    finalize_dur = time.time() - t_fin
                except Exception as e:
                    status = "ERROR"
                    error_msg = str(e)
                    print(f"Batch-level error: {e}", file=sys.stderr)

                chunk_count, entity_count, rel_count = await get_graph_counts(rag)
                metrics = get_system_metrics()
                total_dur = ingest_dur + finalize_dur

                append_csv(batch_log, BATCH_CSV_HEADERS, {
                    "mode": "cumulative",
                    "trial": trial,
                    "batch_size": batch_size,
                    "files_in_batch": len(increment_paths),
                    "cumulative_files": batch_size,
                    "llm_model": args.llm_model,
                    "embed_model": args.embed_model,
                    "ingest_duration_sec": round(ingest_dur, 3),
                    "finalize_duration_sec": round(finalize_dur, 3),
                    "total_duration_sec": round(total_dur, 3),
                    "avg_sec_per_file": round(ingest_dur / len(increment_paths), 3) if increment_paths else 0,
                    "nodes_created": nodes,
                    "relationships_created": rels,
                    "chunk_count": chunk_count,
                    "entity_count": entity_count,
                    "relationship_count": rel_count,
                    "cpu_util_percent": metrics["cpu_util_percent"],
                    "ram_used_gb": metrics["ram_used_gb"],
                    "gpu_util_percent": metrics["gpu_util_percent"],
                    "vram_used_gb": metrics["vram_used_gb"],
                    "failed_files": failed,
                    "status": status,
                    "error_message": error_msg,
                })
                print(f"Checkpoint complete: ingest={ingest_dur:.2f}s finalize={finalize_dur:.2f}s "
                      f"total={total_dur:.2f}s failed={failed}")
                prev_n = batch_size

    print(f"\nCumulative sweep complete. Batch log: {batch_log} | File log: {file_log}")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Batch-Size Ingestion Timing Experiment")
    parser.add_argument("mode", choices=["isolated", "cumulative"], help="Experiment design to run")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10, 25, 50],
                         help="Batch sizes to test (isolated) or cumulative checkpoints (cumulative)")
    parser.add_argument("--trials", type=int, default=1,
                         help="Repeat each batch size this many times to smooth out timing noise")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic file sampling")
    parser.add_argument("--file-timeout", type=float, default=600.0,
                         help="Max seconds to wait for a single file's ingest() call before marking it TIMEOUT "
                              "and moving on (guards against a stuck LLM backend hanging the whole sweep)")
    parser.add_argument("--concurrency", type=int, default=1,
                         help="Per-document chunk-extraction concurrency passed explicitly to "
                              "GraphExtraction (the SDK defaults this to 12 when unset, which caused severe "
                              "contention on local single-GPU inference — tune carefully, don't assume higher "
                              "is always faster)")
    parser.add_argument("--zotero-root", type=str, default=None,
                         help="Zotero data directory (auto-detected under ~/Zotero if omitted)")
    parser.add_argument("--host", type=str, default="localhost", help="FalkorDB host")
    parser.add_argument("--port", type=int, default=6379, help="FalkorDB port")
    parser.add_argument("--password", default=os.getenv("FALKOR_PASSWORD"), help="FalkorDB password")
    parser.add_argument("--graph-name", type=str, default=None,
                         help="Dedicated experiment graph name (defaults to experiments/BatchSize<Mode>)")
    parser.add_argument("--log-file", type=str, default=None, help="Batch-level CSV output path")
    parser.add_argument("--file-log", type=str, default=None, help="Per-file CSV output path")
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL,
                         help="LiteLLM model string for extraction (default: LM Studio's gemma-4-26b-a4b)")
    parser.add_argument("--llm-api-base", type=str, default=LLM_API_BASE, help="LLM server base URL")
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("LLM_API_KEY", "not-needed"),
                         help="API key for the LLM server (local servers usually ignore this)")
    parser.add_argument("--embed-model", type=str, default=EMBEDDING_MODEL,
                         help="LiteLLM model string for embeddings (default: LM Studio's nomic-embed-text-v1.5)")
    parser.add_argument("--embed-api-base", type=str, default=EMBEDDING_API_BASE, help="Embedder server base URL")
    parser.add_argument("--embed-api-key", type=str, default=os.getenv("LLM_API_KEY", "not-needed"),
                         help="API key for the embedder server (local servers usually ignore this)")
    args = parser.parse_args()

    zotero_root = args.zotero_root or find_zotero_root()
    print(f"Using Zotero root: {zotero_root}")
    all_paths = get_all_pdf_paths(zotero_root)
    if not all_paths:
        print("No PDFs found in Zotero library.", file=sys.stderr)
        sys.exit(1)

    max_needed = max(args.sizes)
    if max_needed > len(all_paths):
        print(f"Error: requested batch size {max_needed} exceeds available PDFs ({len(all_paths)}).",
              file=sys.stderr)
        sys.exit(1)

    if args.mode == "isolated":
        asyncio.run(run_isolated(args, all_paths))
    else:
        asyncio.run(run_cumulative(args, all_paths))


if __name__ == "__main__":
    main()
