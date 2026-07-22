"""End-to-end reproduction pipeline for the batch-size ingestion experiment.

Running this one script reproduces the whole study from a clean slate:

  1. Preflight checks    - Zotero library found, FalkorDB reachable+authed,
                            LLM/embedder backend reachable.
  2. Isolated sweep      - run_batch_size_experiment.py isolated
  3. Cumulative sweep    - run_batch_size_experiment.py cumulative
  4. Summarize           - summarize_batch_size_results.py

Every parameter (sizes, trials, seed, model/backend config, Zotero root) and
enough environment detail to reproduce the run (git commit, package
versions, timestamps, per-stage wall-clock time) is written to
results/run_manifest.json, so a given results/batch_size_*.csv can always be
traced back to exactly how it was produced.

Usage:
    FALKOR_PASSWORD=... python experiments/reproduce_batch_size_experiment.py \\
        --sizes 1 5 10 25 50 --trials 3

    # Re-run only the summarizer against existing CSVs:
    python experiments/reproduce_batch_size_experiment.py --skip-isolated --skip-cumulative
"""

import argparse
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from zotero_source import find_zotero_root, get_all_pdf_paths

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# See run_batch_size_experiment.py for backend rationale.
DEFAULT_LLM_MODEL = "ollama/gemma3:latest"
DEFAULT_LLM_API_BASE = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "ollama/nomic-embed-text:latest"
DEFAULT_EMBED_API_BASE = "http://localhost:11434"


def check_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_falkordb(host: str, port: int, password: str | None) -> tuple[bool, str]:
    if not check_tcp(host, port):
        return False, f"Cannot open TCP connection to {host}:{port}"
    try:
        import falkordb
        db = falkordb.FalkorDB(host=host, port=port, password=password, socket_connect_timeout=3)
        if db.connection.ping():
            return True, "OK"
        return False, "Ping returned falsy"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_http_endpoint(api_base: str) -> tuple[bool, str]:
    url = api_base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status < 500, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # A 4xx still proves the server is up and speaking HTTP.
        return e.code < 500, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_preflight(args) -> dict:
    print("=== Preflight checks ===")
    results = {}

    zotero_root = args.zotero_root or find_zotero_root()
    all_paths = get_all_pdf_paths(zotero_root)
    results["zotero_root"] = zotero_root
    results["zotero_pdf_count"] = len(all_paths)
    print(f"  Zotero root: {zotero_root} ({len(all_paths)} PDFs found)")
    if not all_paths:
        print("  FAIL: no PDFs found", file=sys.stderr)
        sys.exit(1)

    max_needed = max(args.sizes)
    if max_needed > len(all_paths):
        print(f"  FAIL: requested batch size {max_needed} exceeds available PDFs ({len(all_paths)})",
              file=sys.stderr)
        sys.exit(1)

    ok, detail = check_falkordb(args.host, args.port, args.password)
    results["falkordb_ok"] = ok
    results["falkordb_detail"] = detail
    print(f"  FalkorDB {args.host}:{args.port}: {'OK' if ok else 'FAIL'} ({detail})")
    if not ok:
        print("  FAIL: cannot reach/authenticate to FalkorDB. Check --host/--port/--password "
              "(or FALKOR_PASSWORD env var).", file=sys.stderr)
        sys.exit(1)

    ok, detail = check_http_endpoint(args.llm_api_base)
    results["llm_backend_ok"] = ok
    results["llm_backend_detail"] = detail
    print(f"  LLM backend {args.llm_api_base}: {'OK' if ok else 'WARN'} ({detail})")
    if not ok:
        print(f"  WARNING: LLM backend at {args.llm_api_base} did not respond cleanly. "
              f"Ingestion will likely fail or hang — verify it's running before proceeding.",
              file=sys.stderr)

    if args.embed_api_base != args.llm_api_base:
        ok, detail = check_http_endpoint(args.embed_api_base)
        results["embed_backend_ok"] = ok
        results["embed_backend_detail"] = detail
        print(f"  Embedder backend {args.embed_api_base}: {'OK' if ok else 'WARN'} ({detail})")

    print()
    return results


def package_versions() -> dict:
    names = ["graphrag-sdk", "litellm", "falkordb", "pypdf", "psutil"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def run_stage(name: str, cmd: list[str]) -> dict:
    print(f"=== {name} ===")
    print(f"$ {' '.join(cmd)}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    duration = time.time() - t0
    status = "SUCCESS" if proc.returncode == 0 else "FAILED"
    print(f"\n{name} {status} in {duration:.1f}s (exit code {proc.returncode})\n")
    return {"status": status, "returncode": proc.returncode, "duration_sec": round(duration, 1)}


def main():
    parser = argparse.ArgumentParser(description="Reproduce the full batch-size ingestion experiment")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10, 25, 50],
                         help="Batch sizes / cumulative checkpoints to run")
    parser.add_argument("--trials", type=int, default=1, help="Repeats per batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic file sampling")
    parser.add_argument("--file-timeout", type=float, default=600.0,
                         help="Max seconds per file's ingest() call before marking it TIMEOUT")
    parser.add_argument("--concurrency", type=int, default=1,
                         help="Per-document chunk-extraction concurrency (see run_batch_size_experiment.py)")
    parser.add_argument("--zotero-root", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--password", default=os.getenv("FALKOR_PASSWORD"))
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-api-base", type=str, default=DEFAULT_LLM_API_BASE)
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("LLM_API_KEY", "not-needed"))
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embed-api-base", type=str, default=DEFAULT_EMBED_API_BASE)
    parser.add_argument("--embed-api-key", type=str, default=os.getenv("LLM_API_KEY", "not-needed"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--skip-isolated", action="store_true")
    parser.add_argument("--skip-cumulative", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    run_started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_started_at": run_started_at,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "config": {
            "sizes": args.sizes,
            "trials": args.trials,
            "seed": args.seed,
            "file_timeout": args.file_timeout,
            "concurrency": args.concurrency,
            "host": args.host,
            "port": args.port,
            "llm_model": args.llm_model,
            "llm_api_base": args.llm_api_base,
            "embed_model": args.embed_model,
            "embed_api_base": args.embed_api_base,
        },
        "stages": {},
    }

    if not args.skip_preflight:
        manifest["preflight"] = run_preflight(args)

    common_flags = [
        "--sizes", *[str(s) for s in args.sizes],
        "--trials", str(args.trials),
        "--seed", str(args.seed),
        "--file-timeout", str(args.file_timeout),
        "--concurrency", str(args.concurrency),
        "--host", args.host,
        "--port", str(args.port),
        "--llm-model", args.llm_model,
        "--llm-api-base", args.llm_api_base,
        "--llm-api-key", args.llm_api_key,
        "--embed-model", args.embed_model,
        "--embed-api-base", args.embed_api_base,
        "--embed-api-key", args.embed_api_key,
    ]
    if args.password:
        common_flags += ["--password", args.password]
    if args.zotero_root:
        common_flags += ["--zotero-root", args.zotero_root]

    python = sys.executable
    runner = os.path.join(THIS_DIR, "run_batch_size_experiment.py")
    summarizer = os.path.join(THIS_DIR, "summarize_batch_size_results.py")

    if not args.skip_isolated:
        manifest["stages"]["isolated"] = run_stage(
            "Isolated sweep", [python, runner, "isolated", *common_flags],
        )

    if not args.skip_cumulative:
        manifest["stages"]["cumulative"] = run_stage(
            "Cumulative sweep", [python, runner, "cumulative", *common_flags],
        )

    if not args.skip_summary:
        manifest["stages"]["summary"] = run_stage(
            "Summarize results", [python, summarizer],
        )

    manifest["run_finished_at"] = datetime.now(timezone.utc).isoformat()

    manifest_path = os.path.join(args.results_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest written to {manifest_path}")

    failed = [name for name, stage in manifest["stages"].items() if stage["status"] != "SUCCESS"]
    if failed:
        print(f"\nPipeline finished with failures in: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
