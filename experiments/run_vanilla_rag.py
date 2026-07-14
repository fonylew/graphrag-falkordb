import os
import time
import argparse
import json
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from litellm import completion

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration constants
OLLAMA_API_BASE = "http://localhost:1234/v1"
LLM_MODEL = "openai/google/gemma-4-26b-a4b"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

SUBSET_PATHS = {
    "medical": {
        "corpus": "/Users/aimet/GraphRAG-Benchmark/Datasets/Corpus/medical.parquet",
        "questions": "/Users/aimet/GraphRAG-Benchmark/Datasets/Questions/medical_questions.parquet",
    },
    "novel": {
        "corpus": "/Users/aimet/GraphRAG-Benchmark/Datasets/Corpus/novel.parquet",
        "questions": "/Users/aimet/GraphRAG-Benchmark/Datasets/Questions/novel_questions.parquet",
    }
}

class SimpleVectorDB:
    def __init__(self):
        self.embeddings = []
        self.texts = []

    def add(self, text, embedding):
        self.texts.append(text)
        self.embeddings.append(embedding)

    def search(self, query_emb, top_k=5):
        if not self.embeddings:
            return []
        # Calculate cosine similarity
        embs = np.array(self.embeddings)
        q_emb = np.array(query_emb)
        
        # Normalize vectors
        norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        norm_q = q_emb / np.linalg.norm(q_emb)
        
        similarities = np.dot(norm_embs, norm_q)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.texts[i] for i in top_indices]

def get_embedding(text):
    import urllib.request
    import json
    url = f"{OLLAMA_API_BASE}/embeddings"
    data = json.dumps({
        "model": EMBED_MODEL,
        "input": text
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res["data"][0]["embedding"]

def chunk_text(text, chunk_size=1200, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla Vector RAG Baseline for GraphRAG-Bench")
    parser.add_argument("--subset", choices=["medical", "novel"], required=True, help="Dataset subset")
    parser.add_argument("--sample", action="store_true", help="Run in sample mode (1 doc, 10k chars max)")
    parser.add_argument("--sample-queries", type=int, default=None, help="Limit number of queries")
    args = parser.parse_args()

    paths = SUBSET_PATHS[args.subset]
    
    # Load corpus
    logging.info(f"Loading corpus from {paths['corpus']}...")
    corpus_df = pd.read_parquet(paths['corpus'])
    corpus_data = corpus_df.to_dict(orient="records")
    
    # Load questions
    logging.info(f"Loading questions from {paths['questions']}...")
    question_df = pd.read_parquet(paths['questions'])
    question_data = question_df.to_dict(orient="records")

    if args.sample:
        corpus_data = corpus_data[:1]
        corpus_data[0]["context"] = corpus_data[0]["context"][:10000]
        question_data = [q for q in question_data if q["source"] == corpus_data[0]["corpus_name"]]
        logging.info(f"Sample Mode active: Truncated context to 10k characters, matching queries: {len(question_data)}.")

    if args.sample_queries:
        question_data = question_data[:args.sample_queries]

    # --- Start Vector Indexing ---
    logging.info("--- Starting Vector Indexing ---")
    start_index_time = time.time()
    db = SimpleVectorDB()
    
    for doc in corpus_data:
        text = doc["context"]
        chunks = chunk_text(text)
        for chunk in chunks:
            emb = get_embedding(chunk)
            db.add(chunk, emb)

    indexing_duration = time.time() - start_index_time
    logging.info(f"✓ Vector indexing complete in {indexing_duration:.2f} seconds.")

    # --- Start Querying Loop ---
    logging.info(f"--- Answering {len(question_data)} questions ---")
    predictions = []
    start_query_time = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def answer_question(q):
        try:
            # Retrieve context
            q_emb = get_embedding(q["question"])
            retrieved_chunks = db.search(q_emb, top_k=5)
            context_str = "\n---\n".join(retrieved_chunks)

            # Generate answer
            prompt = f"""Use the following retrieved passages to answer the user question.
Passages:
{context_str}

Question: {q["question"]}
Answer:"""

            response = completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                api_base=OLLAMA_API_BASE,
                api_key="lm-studio",
                temperature=0.0
            )
            answer = response.choices[0].message.content
            return {
                "id": q["id"],
                "question": q["question"],
                "source": q["source"],
                "context": retrieved_chunks,
                "evidence": q["evidence"],
                "question_type": q["question_type"],
                "generated_answer": answer,
                "ground_truth": q["answer"]
            }
        except Exception as e:
            logging.error(f"Error answering question {q['id']}: {e}")
            return None

    # Run in parallel using a ThreadPoolExecutor
    max_workers = 10  # Concurrency limit for LM Studio
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(answer_question, q): q for q in question_data}
        for future in tqdm(as_completed(futures), total=len(question_data), desc="Querying Vanilla RAG"):
            res = future.result()
            if res:
                predictions.append(res)

    query_duration = time.time() - start_query_time
    avg_query_time = query_duration / len(question_data) if question_data else 0

    # Save output
    output_dir = "results/vanilla_rag"
    os.makedirs(output_dir, exist_ok=True)
    
    pred_path = os.path.join(output_dir, f"predictions_{args.subset}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logging.info(f"✓ Predictions saved to: {pred_path}")

    # Save timings metadata
    meta_path = os.path.join(output_dir, f"predictions_{args.subset}_metadata.json")
    metadata = {
        "subset": args.subset,
        "chunk_size": 1200,
        "chunk_overlap": 100,
        "indexing_time_sec": indexing_duration,
        "total_query_time_sec": query_duration,
        "average_query_time_sec": avg_query_time,
        "num_questions": len(question_data)
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"✓ Timings saved to: {meta_path}")

if __name__ == "__main__":
    main()
