import asyncio
import sys

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)


async def main():
    # Configure providers matching our ingestion setup
    llm = LiteLLM(
        model="ollama/gemma4:latest",
        api_base="http://localhost:11434",
    )

    embedder = LiteLLMEmbedder(
        model="ollama/nomic-embed-text:latest",
        api_base="http://localhost:11434",
    )

    graph_name = "fonylew/GraphRAG"
    print(f"Connecting to GraphRAG on FalkorDB collection: {graph_name}...")
    
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

    # We will query the graph about the ingested paper: "Realizing Memory-Optimized Distributed Graph Processing"
    question = "What is the main topic of the paper 'Realizing Memory-Optimized Distributed Graph Processing' and what are its key contributions?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    print(f"\nQuerying graph with question: '{question}'...\n")

    try:
        async with rag:
            # Query the graph
            response = await rag.completion(question, return_context=True)
            print("=== ANSWER ===")
            print(response.answer)
            print("\n" + "="*30)
            
            # Print the retrieved context items
            print(f"\nRetrieved {len(response.retriever_result.items)} context items:")
            for i, item in enumerate(response.retriever_result.items[:5]):
                score = item.score if item.score is not None else 0.0
                print(f"  [{i+1}] (score={score:.3f}) {item.content[:150]}...")
                
    except Exception as e:
        print(f"Error querying GraphRAG: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
