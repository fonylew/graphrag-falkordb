import asyncio
import os
import shutil
import sqlite3
import sys
import argparse
from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking

# Configuration
GRAPH_NAME = "fonylew/GraphRAG"
OLLAMA_API_BASE = "http://localhost:11434"
LLM_MODEL = "ollama/gemma4:latest"
EMBEDDING_MODEL = "ollama/nomic-embed-text:latest"
EMBEDDING_DIM = 768

def get_zotero_pdf_paths() -> list[str]:
    """Retrieve absolute file paths for PDFs in Zotero collection 37 (GraphRAG)."""
    db_path = "/home/fony/Zotero/zotero.sqlite"
    copy_path = "/home/fony/graphrag-falkordb/zotero_copy.sqlite"

    if not os.path.exists(db_path):
        print(f"Error: Zotero database not found at {db_path}", file=sys.stderr)
        return []

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

        return pdf_paths

    except Exception as e:
        print(f"Error reading Zotero SQLite database: {e}", file=sys.stderr)
        return []
    finally:
        if os.path.exists(copy_path):
            os.remove(copy_path)

def get_rag_instance():
    """Create and return a configured GraphRAG instance."""
    llm = LiteLLM(
        model=LLM_MODEL,
        api_base=OLLAMA_API_BASE,
    )
    embedder = LiteLLMEmbedder(
        model=EMBEDDING_MODEL,
        api_base=OLLAMA_API_BASE,
    )
    return GraphRAG(
        connection=ConnectionConfig(
            host="localhost",
            port=6379,
            graph_name=GRAPH_NAME,
        ),
        llm=llm,
        embedder=embedder,
        embedding_dimension=EMBEDDING_DIM,
    )

async def cmd_status(args):
    """Display status of the GraphRAG collection and Zotero PDFs."""
    rag = get_rag_instance()
    
    print(f"Connecting to FalkorDB collection '{GRAPH_NAME}'...")
    async with rag:
        try:
            # Query ingested documents
            doc_res = await rag._conn.query("MATCH (d:Document) RETURN d.id")
            ingested_docs = {row[0] for row in doc_res.result_set}
        except Exception as e:
            print(f"Error querying FalkorDB: {e}", file=sys.stderr)
            return

        # Query other stats
        chunk_res = await rag._conn.query("MATCH (c:Chunk) RETURN count(c)")
        chunk_count = chunk_res.result_set[0][0] if chunk_res.result_set else 0
        
        entity_res = await rag._conn.query("MATCH (e:__Entity__) RETURN count(e)")
        entity_count = entity_res.result_set[0][0] if entity_res.result_set else 0
        
        rel_res = await rag._conn.query("MATCH ()-[r:RELATES]->() RETURN count(r)")
        rel_count = rel_res.result_set[0][0] if rel_res.result_set else 0

    print("\n=== GraphRAG Database Statistics ===")
    print(f"Total Ingested Documents: {len(ingested_docs)}")
    print(f"Total Chunks:            {chunk_count}")
    print(f"Total Entities:          {entity_count}")
    print(f"Total Relationships:     {rel_count}")

    print("\n=== Zotero Collection Status ===")
    zotero_pdfs = get_zotero_pdf_paths()
    if not zotero_pdfs:
        print("No PDFs found in Zotero Collection 37.")
        return

    for idx, path in enumerate(zotero_pdfs):
        filename = os.path.basename(path)
        status = "[INGESTED]" if path in ingested_docs else "[PENDING]"
        print(f"  {idx+1:02d}. {status:<10} {filename}")

async def cmd_ingest(args):
    """Ingest new/pending documents into the GraphRAG collection."""
    zotero_pdfs = get_zotero_pdf_paths()
    if not zotero_pdfs:
        print("No files to process.")
        return

    rag = get_rag_instance()
    
    async with rag:
        # Get already ingested documents to determine pending ones
        try:
            doc_res = await rag._conn.query("MATCH (d:Document) RETURN d.id")
            ingested_docs = {row[0] for row in doc_res.result_set}
        except Exception as e:
            print(f"Error checking existing documents: {e}", file=sys.stderr)
            return

        to_ingest = []
        for path in zotero_pdfs:
            if args.file and path != args.file and os.path.basename(path) != args.file:
                continue
            
            is_ingested = path in ingested_docs
            if is_ingested and not args.force:
                if args.file:
                    print(f"File '{os.path.basename(path)}' is already ingested. Use --force to re-ingest.")
                continue
            
            to_ingest.append(path)

        if args.file and not to_ingest:
            print(f"File '{args.file}' not found in Zotero Collection 37 or already ingested.")
            return

        if not args.file and not args.all:
            # By default, only ingest pending ones
            to_ingest = [p for p in to_ingest if p not in ingested_docs]

        if args.limit and args.limit > 0:
            to_ingest = to_ingest[:args.limit]

        if not to_ingest:
            print("No pending documents to ingest.")
            return

        print(f"\nPreparing to ingest {len(to_ingest)} document(s):")
        for path in to_ingest:
            print(f"  - {os.path.basename(path)}")

        if args.dry_run:
            print("\nDry-run mode: Exiting without changes.")
            return

        # Perform deletions for forced items first
        for path in to_ingest:
            if path in ingested_docs:
                print(f"Deleting existing document from graph first: {os.path.basename(path)}")
                await rag.delete_document(path, if_missing="ignore")

        # Start Ingestion
        print(f"\nStarting ingestion of {len(to_ingest)} PDF(s) into collection '{GRAPH_NAME}'...")
        print("Using chunk_size=4000, max_concurrency=1 for local Ollama stability.")
        
        try:
            results = await rag.ingest(
                to_ingest,
                chunker=FixedSizeChunking(chunk_size=4000, chunk_overlap=500),
                max_concurrency=1,
            )
            
            # Print ingestion results
            if isinstance(results, list):
                for path, res in zip(to_ingest, results):
                    filename = os.path.basename(path)
                    if isinstance(res, Exception):
                        print(f"  - Failed '{filename}': {res}")
                    else:
                        print(f"  - Ingested '{filename}': {res.nodes_created} nodes, {res.relationships_created} edges created.")
            else:
                print(f"  - Ingestion complete. Nodes: {results.nodes_created}, relationships: {results.relationships_created}")

            print("\nFinalizing GraphRAG index (deduplication & vector indexing)...")
            await rag.finalize()
            print("Successfully completed ingestion and finalized the index!")

        except Exception as e:
            print(f"\nAn error occurred during ingestion: {e}", file=sys.stderr)
            raise

async def cmd_query(args):
    """Query the GraphRAG collection."""
    question = args.question
    rag = get_rag_instance()
    
    print(f"Connecting to collection '{GRAPH_NAME}'...")
    print(f"Querying with: '{question}'...\n")
    
    try:
        async with rag:
            response = await rag.completion(question, return_context=True)
            print("=== ANSWER ===")
            print(response.answer)
            print("\n" + "="*30)
            
            print(f"\nRetrieved {len(response.retriever_result.items)} context items:")
            for i, item in enumerate(response.retriever_result.items[:5]):
                score = item.score if item.score is not None else 0.0
                print(f"  [{i+1}] (score={score:.3f}) {item.content[:150]}...")
    except Exception as e:
        print(f"Error querying GraphRAG: {e}", file=sys.stderr)

async def cmd_clear(args):
    """Clear all data in the GraphRAG collection."""
    confirm = input(f"Are you sure you want to delete ALL data in the '{GRAPH_NAME}' collection? (y/N): ")
    if confirm.lower() != 'y':
        print("Clear canceled.")
        return
        
    rag = get_rag_instance()
    print(f"Deleting collection '{GRAPH_NAME}'...")
    async with rag:
        try:
            await rag.delete_all()
            print("Collection cleared successfully.")
        except Exception as e:
            print(f"Error clearing collection: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="GraphRAG FalkorDB Collection Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status command
    subparsers.add_parser("status", aliases=["list"], help="Show database stats and Zotero PDF ingestion status")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from Zotero")
    ingest_parser.add_argument("--all", action="store_true", help="Ingest all files, including already ingested ones (unless --force is omitted)")
    ingest_parser.add_argument("--limit", type=int, help="Limit the number of documents to ingest")
    ingest_parser.add_argument("--file", type=str, help="Ingest a specific file by filename or absolute path")
    ingest_parser.add_argument("--force", action="store_true", help="Re-ingest and overwrite already ingested documents")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Show files to ingest without making changes")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the GraphRAG collection")
    query_parser.add_argument("question", type=str, help="The query question")

    # Clear command
    subparsers.add_parser("clear", help="Delete all data in the collection")

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    if args.command in ["status", "list"]:
        loop.run_until_complete(cmd_status(args))
    elif args.command == "ingest":
        loop.run_until_complete(cmd_ingest(args))
    elif args.command == "query":
        loop.run_until_complete(cmd_query(args))
    elif args.command == "clear":
        loop.run_until_complete(cmd_clear(args))

if __name__ == "__main__":
    main()
