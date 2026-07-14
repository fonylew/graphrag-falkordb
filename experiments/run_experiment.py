import asyncio
import os
import sys
import time
import csv
import argparse
import random
import psutil
import shutil
import subprocess
from datetime import datetime

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking

# Default Constants
GRAPH_NAME = "fonylew/GraphRAG_Experiment"
OLLAMA_API_BASE = "http://localhost:11434"
LLM_MODEL = "ollama/gemma4:latest"
EMBEDDING_MODEL = "ollama/nomic-embed-text:latest"
EMBEDDING_DIM = 768

def get_gpu_metrics():
    """Retrieve GPU utilization and VRAM usage if nvidia-smi is available."""
    try:
        # Run nvidia-smi command to get GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("\n")[0].split(",")
            gpu_util = float(parts[0].strip())
            vram_used = float(parts[1].strip()) / 1024.0  # Convert to GB
            return gpu_util, vram_used
    except Exception:
        pass
    return 0.0, 0.0

def get_system_metrics():
    """Collect current CPU, RAM, and GPU usage metrics."""
    cpu_util = psutil.cpu_percent()
    ram_util = psutil.virtual_memory().percent
    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
    gpu_util, vram_used_gb = get_gpu_metrics()
    
    return {
        "cpu_util_percent": cpu_util,
        "ram_util_percent": ram_util,
        "ram_used_gb": round(ram_used_gb, 2),
        "gpu_util_percent": gpu_util,
        "vram_used_gb": round(vram_used_gb, 2)
    }

def generate_mock_datasets(raw_dir: str, env_dir: str, count: int):
    """Generate mock text files with structured facts if directories are empty."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(env_dir, exist_ok=True)
    
    # Check if there are already files in the raw folder
    existing_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    if len(existing_files) >= count:
        print(f"Dataset already exists with {len(existing_files)} files. Skipping mock generation.")
        return

    print(f"Generating {count} mock text files in {raw_dir} for the experiment...")
    first_names = ["John", "Sarah", "Emily", "Michael", "David", "Jessica", "Robert", "James", "Karen", "Thomas"]
    last_names = ["Smith", "Doe", "Johnson", "Miller", "Williams", "Brown", "Jones", "Davis", "Wilson", "Taylor"]
    companies = ["AcmeCorp", "Globex", "Initech", "UmbrellaCorp", "Cyberdyne", "Soylent", "Hooli", "Vehement", "StarkInd", "WayneEnt"]
    cities = ["London", "New York", "Berlin", "Tokyo", "Paris", "Rome", "Sydney", "Toronto", "Austin", "Seattle"]
    roles = ["Software Engineer", "VP of Engineering", "CEO", "Data Scientist", "Product Manager", "Systems Architect"]

    for i in range(count):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        comp = random.choice(companies)
        city = random.choice(cities)
        role = random.choice(roles)
        
        content = (
            f"{fn} {ln} is a {role} working at {comp} based in {city}.\n"
            f"The company {comp} specializes in advanced solutions and was founded in {random.randint(1990, 2024)}.\n"
            f"In {city}, {fn} collaborates closely with regional partners to scale operations.\n"
        )
        
        file_path = os.path.join(raw_dir, f"doc_{i+1:03d}.txt")
        with open(file_path, "w") as f:
            f.write(content)
            
    print("Mock dataset generation complete.")

async def run_scenario(args):
    # Establish folder paths
    raw_dir = os.path.abspath(args.raw_dir)
    env_dir = os.path.abspath(args.env_dir)
    results_dir = os.path.dirname(os.path.abspath(args.log_file))
    os.makedirs(results_dir, exist_ok=True)
    
    # Auto-generate mock data if folders are empty
    generate_mock_datasets(raw_dir, env_dir, args.count)
    
    # Gather file pool from raw_dir
    all_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".txt")])
    if not all_files:
        print(f"Error: No text files found in {raw_dir}", file=sys.stderr)
        return
        
    print(f"\n--- Starting Experiment Scenario ---")
    print(f"Interval: {args.interval} seconds")
    print(f"Count: {args.count} files")
    print(f"Mode: {args.mode}")
    print(f"Finalize Strategy: {args.finalize}")
    print(f"Log File: {args.log_file}")
    print(f"Graph Collection: {GRAPH_NAME}")
    
    # Initialize RAG configurations
    llm = LiteLLM(model=LLM_MODEL, api_base=OLLAMA_API_BASE)
    embedder = LiteLLMEmbedder(model=EMBEDDING_MODEL, api_base=OLLAMA_API_BASE)
    
    rag = GraphRAG(
        connection=ConnectionConfig(
            host=args.host,
            port=args.port,
            password=args.password,
            graph_name=GRAPH_NAME,
        ),
        llm=llm,
        embedder=embedder,
        embedding_dimension=EMBEDDING_DIM,
    )
    
    # Initialize CSV logger
    csv_headers = [
        "timestamp", "cycle", "file_name", "operation", "status", 
        "ingest_duration_sec", "finalize_duration_sec", "total_duration_sec",
        "cpu_util_percent", "ram_util_percent", "ram_used_gb", 
        "gpu_util_percent", "vram_used_gb", "queue_backlog_count", "error_message"
    ]
    
    with open(args.log_file, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(csv_headers)
        
    queue = []
    
    async with rag:
        # Clear collection if requested at start of trial
        if args.clear_start:
            print("Clearing GraphRAG collection before starting...")
            await rag.delete_all()
            
        for cycle in range(1, args.count + 1):
            cycle_start = time.time()
            timestamp_str = datetime.now().isoformat()
            
            # Prepare file for this cycle
            file_name = all_files[(cycle - 1) % len(all_files)]
            source_path = os.path.join(raw_dir, file_name)
            target_path = os.path.join(env_dir, file_name)
            
            operation = "ingest"
            if args.mode == "update":
                # Copy file first if not exists
                if not os.path.exists(target_path):
                    shutil.copy2(source_path, target_path)
                    print(f"[Cycle {cycle}] Copied base file {file_name} to test environment.")
                
                # Append modification to simulate knowledge update
                with open(target_path, "a") as f:
                    f.write(f"Update modification at {timestamp_str} (Cycle {cycle}).\n")
                operation = "update"
            else:
                # Mode is 'ingest': copy file fresh
                shutil.copy2(source_path, target_path)
                operation = "ingest"
                
            # Add to run queue
            queue.append(target_path)
            print(f"\n[Cycle {cycle}] Queueing: {file_name} (Queue Size: {len(queue)})")
            
            # Record status before executing
            metrics = get_system_metrics()
            
            # Dequeue and execute
            current_file = queue.pop(0)
            status = "SUCCESS"
            error_msg = ""
            ingest_dur = 0.0
            finalize_dur = 0.0
            
            t0 = time.time()
            try:
                if operation == "update":
                    print(f"Executing rag.update() on {file_name}...")
                    await rag.update(file_path=current_file)
                else:
                    print(f"Executing rag.ingest() on {file_name}...")
                    await rag.ingest([current_file])
                ingest_dur = time.time() - t0
                
                if args.finalize == "immediate":
                    print(f"Executing rag.finalize()...")
                    t_fin = time.time()
                    await rag.finalize()
                    finalize_dur = time.time() - t_fin
                    
            except Exception as e:
                status = "ERROR"
                error_msg = str(e)
                print(f"Error during execution: {error_msg}", file=sys.stderr)
                
            total_dur = time.time() - t0
            
            # Record post-execution status
            sys_metrics = get_system_metrics()
            
            # Save results
            with open(args.log_file, "a", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow([
                    timestamp_str, cycle, file_name, operation, status,
                    round(ingest_dur, 3), round(finalize_dur, 3), round(total_dur, 3),
                    sys_metrics["cpu_util_percent"], sys_metrics["ram_util_percent"], sys_metrics["ram_used_gb"],
                    sys_metrics["gpu_util_percent"], sys_metrics["vram_used_gb"], len(queue), error_msg
                ])
                
            print(f"Cycle {cycle} complete. Ingest: {ingest_dur:.2f}s | Finalize: {finalize_dur:.2f}s | Total: {total_dur:.2f}s")
            
            # Enforce Interval Delay
            elapsed = time.time() - cycle_start
            wait_time = args.interval - elapsed
            if wait_time > 0:
                print(f"Waiting {wait_time:.2f}s until next cycle...")
                await asyncio.sleep(wait_time)
            else:
                backlog = len(queue) + int(abs(wait_time) // args.interval)
                print(f"WARNING: Cycle overran interval by {abs(wait_time):.2f}s! Backlog buildup present.")
                
        # Handle Batched Finalization
        if args.finalize == "batched":
            print("\nExecuting Batched Finalization...")
            t_fin = time.time()
            timestamp_str = datetime.now().isoformat()
            status = "SUCCESS"
            error_msg = ""
            try:
                await rag.finalize()
            except Exception as e:
                status = "ERROR"
                error_msg = str(e)
                print(f"Error during batched finalize: {error_msg}", file=sys.stderr)
            finalize_dur = time.time() - t_fin
            
            sys_metrics = get_system_metrics()
            with open(args.log_file, "a", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow([
                    timestamp_str, "FINAL_BATCH", "N/A", "finalize", status,
                    0.0, round(finalize_dur, 3), round(finalize_dur, 3),
                    sys_metrics["cpu_util_percent"], sys_metrics["ram_util_percent"], sys_metrics["ram_used_gb"],
                    sys_metrics["gpu_util_percent"], sys_metrics["vram_used_gb"], 0, error_msg
                ])
            print(f"Batched Finalization complete in {finalize_dur:.2f}s")

    print(f"\nExperiment complete. Results saved to {args.log_file}")

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Bulk Ingestion Bottleneck Experiment Runner")
    parser.add_argument("--interval", type=int, default=60, help="Ingestion interval in seconds (e.g. 60, 30, 10)")
    parser.add_argument("--count", type=int, default=30, help="Number of files/cycles to run")
    parser.add_argument("--mode", type=str, choices=["ingest", "update"], default="ingest", help="Ingest new files or update existing ones")
    parser.add_argument("--finalize", type=str, choices=["immediate", "batched"], default="immediate", help="Finalization strategy")
    parser.add_argument("--raw-dir", type=str, default="data/raw_dataset", help="Directory with original raw files")
    parser.add_argument("--env-dir", type=str, default="data/test_environment", help="Ingestion watched environment directory")
    parser.add_argument("--log-file", type=str, default="results/experiment_results.csv", help="CSV log output path")
    parser.add_argument("--host", type=str, default="localhost", help="FalkorDB host")
    parser.add_argument("--port", type=int, default=6379, help="FalkorDB port")
    parser.add_argument("--password", default=os.getenv("FALKOR_PASSWORD"), help="FalkorDB password")
    parser.add_argument("--clear-start", action="store_true", help="Clear the database graph before starting")
    
    args = parser.parse_args()
    
    asyncio.run(run_scenario(args))

if __name__ == "__main__":
    main()
