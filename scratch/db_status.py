import asyncio
from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder

async def main():
    llm = LiteLLM(model="ollama/gemma4:latest", api_base="http://localhost:11434")
    embedder = LiteLLMEmbedder(model="ollama/nomic-embed-text:latest", api_base="http://localhost:11434")
    graph_name = "fonylew/GraphRAG"
    
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", port=6379, graph_name=graph_name),
        llm=llm,
        embedder=embedder,
        embedding_dimension=768,
    )
    
    async with rag:
        # Check Document nodes
        doc_query = "MATCH (d:Document) RETURN d.id, d.title"
        docs = await rag._graph_store.query(doc_query)
        print(f"Total documents: {len(docs)}")
        for idx, doc in enumerate(docs):
            print(f"  {idx+1}: {doc[0]}")
            
        # Count other types of nodes
        chunk_count = await rag._graph_store.query("MATCH (c:Chunk) RETURN count(c)")
        entity_count = await rag._graph_store.query("MATCH (e:__Entity__) RETURN count(e)")
        rel_count = await rag._graph_store.query("MATCH ()-[r:RELATES]->() RETURN count(r)")
        
        print(f"Total Chunks: {chunk_count[0][0] if chunk_count else 0}")
        print(f"Total Entities: {entity_count[0][0] if entity_count else 0}")
        print(f"Total Relationships: {rel_count[0][0] if rel_count else 0}")

if __name__ == "__main__":
    asyncio.run(main())
