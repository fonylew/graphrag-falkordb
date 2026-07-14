import asyncio
import os
import sys
import time
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking

# Configuration Defaults
GRAPH_NAME = "fonylew/GraphRAG_Benchmark"
OLLAMA_API_BASE = "http://localhost:1234/v1"
LLM_MODEL = "openai/google/gemma-4-26b-a4b"
EMBEDDING_MODEL = "openai/text-embedding-nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# Default paths in GraphRAG-Benchmark
BENCHMARK_DIR = "/Users/aimet/GraphRAG-Benchmark"
SUBSET_PATHS = {
    "medical": {
        "corpus": os.path.join(BENCHMARK_DIR, "Datasets/Corpus/medical.parquet"),
        "questions": os.path.join(BENCHMARK_DIR, "Datasets/Questions/medical_questions.parquet")
    },
    "novel": {
        "corpus": os.path.join(BENCHMARK_DIR, "Datasets/Corpus/novel.parquet"),
        "questions": os.path.join(BENCHMARK_DIR, "Datasets/Questions/novel_questions.parquet")
    }
}

async def run_ingestion(rag, corpus_data, chunk_size, chunk_overlap):
    """Ingest corpus documents into FalkorDB GraphRAG."""
    print(f"\n--- Starting Ingestion of {len(corpus_data)} document(s) ---")
    t0 = time.time()
    
    # We ingest documents sequentially or concurrently. To prevent overloading Ollama, 
    # we ingest them and then finalize.
    for i, doc in enumerate(corpus_data):
        doc_name = doc["corpus_name"]
        context = doc["context"]
        print(f"[{i+1}/{len(corpus_data)}] Ingesting Document: {doc_name} ({len(context.split())} words)...")
        
        # Ingest text directly
        await rag.ingest(
            text=context,
            document_id=doc_name,
            chunker=FixedSizeChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            max_concurrency=1
        )
        
    print("\nFinalizing index (Entity deduplication and relationship embedding)...")
    t_fin = time.time()
    await rag.finalize()
    finalize_dur = time.time() - t_fin
    
    total_dur = time.time() - t0
    print(f"✓ Ingestion & Finalization complete in {total_dur:.2f} seconds (Finalize took {finalize_dur:.2f}s).")
    
    return {
        "ingest_time_sec": round(total_dur - finalize_dur, 2),
        "finalize_time_sec": round(finalize_dur, 2),
        "total_indexing_time_sec": round(total_dur, 2)
    }

async def run_questions(rag, question_data, output_path, sample_count=None):
    """Answer evaluation questions and save the predictions in standard format."""
    print(f"\n--- Answering {len(question_data)} evaluation questions ---")
    predictions = []
    
    if sample_count and sample_count < len(question_data):
        print(f"Sampling first {sample_count} questions for evaluation.")
        question_data = question_data[:sample_count]
        
    t0 = time.time()
    for i, q in enumerate(tqdm(question_data, desc="Querying GraphRAG")):
        question_id = q["id"]
        question_text = q["question"]
        source = q["source"]
        evidence = q["evidence"]
        question_type = q["question_type"]
        ground_truth = q["answer"]
        
        try:
            # Query the GraphRAG pipeline
            response = await rag.completion(question_text, return_context=True)
            generated_answer = response.answer
            
            # Format retrieved context chunks
            retrieved_chunks = []
            if hasattr(response, 'retriever_result') and response.retriever_result:
                for item in response.retriever_result.items:
                    retrieved_chunks.append(item.content)
            retrieved_context = "\n\n".join(retrieved_chunks)
            
        except Exception as e:
            print(f"\nError answering question {question_id}: {e}", file=sys.stderr)
            generated_answer = "I don't know (Error during query execution)"
            retrieved_context = ""
            
        predictions.append({
            "id": question_id,
            "question": question_text,
            "source": source,
            "context": retrieved_context,
            "evidence": evidence,
            "question_type": question_type,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth
        })
        
    dur = time.time() - t0
    print(f"Querying complete in {dur:.2f}s (Average: {dur/len(predictions):.2f}s per query).")
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"✓ Predictions successfully saved to: {output_path}")
    
    return {
        "total_query_time_sec": round(dur, 2),
        "average_query_time_sec": round(dur / len(predictions), 3),
        "num_questions": len(predictions)
    }

async def main_async(args):
    # Determine Parquet file paths
    paths = SUBSET_PATHS.get(args.subset)
    if not paths:
        print(f"Error: Invalid subset '{args.subset}'", file=sys.stderr)
        return
        
    corpus_path = paths["corpus"]
    questions_path = paths["questions"]
    
    # Check if files exist
    if not os.path.exists(corpus_path) or not os.path.exists(questions_path):
        print(f"Error: Parquet dataset files not found at:\n  - {corpus_path}\n  - {questions_path}", file=sys.stderr)
        print("Please check if the path /Users/aimet/GraphRAG-Benchmark is correct and contains Datasets.", file=sys.stderr)
        return

    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
    corpus_data = []
    for item in corpus_dataset:
        corpus_data.append({
            "corpus_name": item["corpus_name"],
            "context": item["context"]
        })
    print(f"Loaded {len(corpus_data)} documents.")
    
    # Load questions
    print(f"Loading questions from {questions_path}...")
    questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
    question_data = []
    for item in questions_dataset:
        question_data.append({
            "id": item["id"],
            "source": item["source"],
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "evidence": item["evidence"]
        })
    print(f"Loaded {len(question_data)} questions.")

    if args.sample:
        # In sample mode, restrict corpus to 1 document and questions to a small number
        corpus_data = corpus_data[:1]
        corpus_data[0]["context"] = corpus_data[0]["context"][:10000]  # Truncate text for fast sample run!
        question_data = [q for q in question_data if q["source"] == corpus_data[0]["corpus_name"]]
        print(f"Sample Mode active: Ingesting only 1 document (truncated to 10k chars) and querying matching questions ({len(question_data)}).")

    # Connect to FalkorDB GraphRAG
    llm = LiteLLM(model=LLM_MODEL, api_base=OLLAMA_API_BASE)
    embedder = LiteLLMEmbedder(model=EMBEDDING_MODEL, api_base=OLLAMA_API_BASE)
    
    rag = GraphRAG(
        connection=ConnectionConfig(
            host=args.host,
            port=args.port,
            password=args.password,
            graph_name=args.graph_name,
        ),
        llm=llm,
        embedder=embedder,
        embedding_dimension=EMBEDDING_DIM,
    )

    async with rag:
        if args.clear:
            print("Clearing GraphRAG collection before running benchmark...")
            await rag.delete_all()
            
        ingest_metrics = {}
        if args.mode in ["all", "ingest"]:
            ingest_metrics = await run_ingestion(rag, corpus_data, args.chunk_size, args.chunk_overlap)
            
        query_metrics = {}
        if args.mode in ["all", "query"]:
            query_metrics = await run_questions(rag, question_data, args.output_file, sample_count=args.sample_queries)
            
        if ingest_metrics or query_metrics:
            metrics_path = args.output_file.replace(".json", "_metadata.json")
            combined_metrics = {
                "subset": args.subset,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                **ingest_metrics,
                **query_metrics
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(combined_metrics, f, indent=2, ensure_ascii=False)
            print(f"✓ Ingestion & query timings saved to: {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="FalkorDB GraphRAG Benchmark Runner")
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], help="Dataset subset to benchmark")
    parser.add_argument("--mode", default="all", choices=["all", "ingest", "query"], help="Benchmark execution stage")
    parser.add_argument("--graph-name", default=GRAPH_NAME, help="FalkorDB graph collection name")
    parser.add_argument("--host", default="localhost", help="FalkorDB Redis host")
    parser.add_argument("--port", type=int, default=6379, help="FalkorDB Redis port")
    parser.add_argument("--password", default=os.getenv("FALKOR_PASSWORD"), help="FalkorDB Redis password")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Text chunk size for ingestion")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Text chunk overlap size")
    parser.add_argument("--output-file", default="results/falkordb_graphrag/predictions_medical.json", help="Path to save predictions JSON")
    parser.add_argument("--sample", action="store_true", help="Run with a small sample to test integration")
    parser.add_argument("--sample-queries", type=int, default=None, help="Limit number of queries to run")
    parser.add_argument("--clear", action="store_true", help="Clear the database graph before starting")
    
    args = parser.parse_args()
    
    # Auto-adjust output filename if not customized
    if args.output_file == "results/falkordb_graphrag/predictions_medical.json" and args.subset == "novel":
        args.output_file = "results/falkordb_graphrag/predictions_novel.json"
        
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
