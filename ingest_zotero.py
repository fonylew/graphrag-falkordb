import asyncio
import os
import shutil
import sqlite3
import sys

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking


def get_zotero_pdf_paths(limit: int = None) -> list[str]:
    """Retrieve absolute file paths for PDFs in collection 37 (GraphRAG)."""
    db_path = "/home/fony/Zotero/zotero.sqlite"
    copy_path = "/home/fony/graphrag-falkordb/zotero_copy.sqlite"

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        return []

    print(f"Copying Zotero DB to {copy_path}...")
    shutil.copy2(db_path, copy_path)

    try:
        conn = sqlite3.connect(copy_path)
        cursor = conn.cursor()

        # Query to fetch attachments for collection ID 37 (GraphRAG) with application/pdf content type
        query = """
        SELECT 
            attachments.key AS attachmentKey,
            itemAttachments.path
        FROM collectionItems
        JOIN items ON collectionItems.itemID = items.itemID
        LEFT JOIN itemAttachments ON itemAttachments.parentItemID = items.itemID
        LEFT JOIN items AS attachments ON itemAttachments.itemID = attachments.itemID
        WHERE collectionItems.collectionID = 37 AND itemAttachments.contentType = 'application/pdf';
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"Found {len(rows)} PDF attachment records in collection 'GraphRAG' (ID 37).")

        pdf_paths = []
        for row in rows:
            attach_key, rel_path = row
            if not rel_path or not attach_key:
                continue

            if rel_path.startswith("storage:"):
                filename = rel_path[len("storage:") :]
                full_path = f"/home/fony/Zotero/storage/{attach_key}/{filename}"
            else:
                full_path = rel_path

            if os.path.exists(full_path):
                pdf_paths.append(full_path)
            else:
                print(f"Warning: File not found on disk: {full_path}", file=sys.stderr)

        print(f"Verified {len(pdf_paths)} out of {len(rows)} files exist on disk.")
        
        if limit is not None and limit > 0:
            print(f"Applying trial limit: selecting first {limit} files.")
            pdf_paths = pdf_paths[:limit]
            
        return pdf_paths

    except Exception as e:
        print(f"Error reading Zotero SQLite database: {e}", file=sys.stderr)
        return []
    finally:
        if os.path.exists(copy_path):
            os.remove(copy_path)


async def main():
    # Parse command line arguments for a dry-run limit
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"Running with trial limit = {limit}")
        except ValueError:
            pass

    # 1. Fetch PDF files from Zotero
    pdf_paths = get_zotero_pdf_paths(limit=limit)
    if not pdf_paths:
        print("No files to ingest. Exiting.", file=sys.stderr)
        return

    # 2. Configure providers
    print("Configuring Gemma4 LLM (Ollama)...")
    llm = LiteLLM(
        model="ollama/gemma4:latest",
        api_base="http://localhost:11434",
    )

    print("Configuring nomic-embed-text Embedder (Ollama)...")
    embedder = LiteLLMEmbedder(
        model="ollama/nomic-embed-text:latest",
        api_base="http://localhost:11434",
    )

    # 3. Create GraphRAG instance
    graph_name = "fonylew/GraphRAG"
    print(f"Initializing GraphRAG on FalkorDB collection: {graph_name}")
    
    rag = GraphRAG(
        connection=ConnectionConfig(
            host="localhost",
            port=6379,
            graph_name=graph_name,
        ),
        llm=llm,
        embedder=embedder,
        embedding_dimension=768,
    )

    # 4. Ingest PDFs
    print(f"Starting ingestion of {len(pdf_paths)} PDFs into collection '{graph_name}'...")
    print("Using chunk_size=4000 for fast, high-quality, and timeout-free local extraction.")
    print("Using max_concurrency=1 to prevent overloading local LLM services.")
    
    try:
        async with rag:
            # We set max_concurrency=1 here for stability
            # We use FixedSizeChunking with size 4000 and overlap 500 for optimal local performance
            results = await rag.ingest(
                pdf_paths,
                chunker=FixedSizeChunking(chunk_size=4000, chunk_overlap=500),
                max_concurrency=1,
            )
            
            # Print individual document results
            if isinstance(results, list):
                for path, res in zip(pdf_paths, results):
                    filename = os.path.basename(path)
                    if isinstance(res, Exception):
                        print(f"  - Failed to ingest '{filename}': {res}")
                    else:
                        print(f"  - Ingested '{filename}': {res.nodes_created} nodes, {res.relationships_created} edges created.")
            else:
                print(f"  - Ingestion complete. Nodes created: {results.nodes_created}, relationships: {results.relationships_created}")

            # 5. Finalize GraphRAG (dedup + embed entities/relationships + build vector indices)
            print("\nFinalizing GraphRAG index (deduplication and vector indexing)...")
            await rag.finalize()
            print("Deduplication and vector indexing finalized successfully!")

    except Exception as e:
        print(f"\nAn error occurred during GraphRAG process: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    asyncio.run(main())
