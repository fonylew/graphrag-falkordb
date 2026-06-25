# Local GraphRAG Knowledge Base & ADK Agent

A fully local GraphRAG (Graph Retrieval-Augmented Generation) system built using FalkorDB, Ollama, and the Google Agent Development Kit (ADK). 

This repository allows you to ingest research papers (such as PDFs exported from Zotero), extract entities and relationships into a local FalkorDB graph database, and query this knowledge base through an interactive AI research agent running local LLMs (like Gemma 4 or models served via LM Studio / llama.cpp).

---

## Key Features
- **100% Local execution**: No paid API keys or external cloud dependencies are required.
- **GraphRAG pipeline**: Connects the FalkorDB graph database with LLM-based entity-relationship extraction and retrieval pipelines.
- **Interactive ADK Agent**: A conversational agent built with Google ADK, providing a uvicorn-based FastAPI backend and an interactive Web UI playground.
- **Hardware Acceleration**: Support for NVIDIA GPU (CUDA) acceleration for both local compilation and model inference.
- **Containerized Deployments**: Easy deployment on local or remote machines using Docker and Docker Compose.

---

## Repository Structure
- **[manage_rag.py](file:///home/fony/graphrag-falkordb/manage_rag.py)**: Central command-line tool to inspect FalkorDB status, ingest PDFs, query RAG, or clear data.
- **[graphrag_agent/](file:///home/fony/graphrag-falkordb/graphrag_agent/)**: The Google ADK agent service.
  - `app/agent.py`: Agent definition containing the `query_graphrag` tool and model configurations.
  - `app/fast_api_app.py`: FastAPI server configuration for telemetry and endpoints.
  - `Dockerfile`: Multi-stage build based on `python:3.11-slim` for lightweight deployment.
- **[docker-compose.yml](file:///home/fony/graphrag-falkordb/docker-compose.yml)**: Service orchestration mapping FalkorDB, Ollama, llama.cpp, and the Agent container.
- **[DEPLOY.md](file:///home/fony/graphrag-falkordb/DEPLOY.md)**: Exhaustive deployment guide for target servers, GPU configs, and local LLM options.
- **[graphrag_sdk_docs.md](file:///home/fony/graphrag-falkordb/graphrag_sdk_docs.md)**: In-depth technical documentation covering the FalkorDB GraphRAG SDK ingestion and query architecture.

---

## Local Quickstart (Host Machine)

### Prerequisites
*(Note: If you prefer to run Ollama, FalkorDB, and the Agent entirely in Docker containers using Docker Compose, skip this local setup and refer directly to [DEPLOY.md](file:///home/fony/graphrag-falkordb/DEPLOY.md) or the **Docker Compose Deployment** section below).*

1. **Local LLM Runner**:
   * **Option A**: **Ollama** running locally on port `11434` with the following models downloaded:
     ```bash
     ollama pull gemma4:latest
     ollama pull nomic-embed-text:latest
     ```
   * **Option B**: **LM Studio** or a **llama.cpp** server running and serving an OpenAI-compatible API on your desired port (e.g. `1234` or `8080`).
2. **FalkorDB** running via Docker:
   ```bash
   docker run -p 6379:6379 -it --rm falkordb/falkordb:edge
   ```

### 1. Set Up Virtual Environment
Configure a local Python 3.11 environment (to utilize prebuilt dependencies without manual compilation) and install requirements:
```bash
# Create Python 3.11 environment
uv venv -p 3.11

# Sync dependencies (pointing to PyTorch CUDA index if using NVIDIA GPU)
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 uv sync
```

### 2. Ingest Data
Place your research PDFs inside a folder and run the ingestion pipeline to parse text, build the graph, and write to FalkorDB:
```bash
uv run python manage_rag.py ingest --path /path/to/papers/
```
To check status or clean up the database:
```bash
# Check database ingestion and entity count
uv run python manage_rag.py status

# Clear all entities from the graph database
uv run python manage_rag.py clear
```

### 3. Run the ADK Agent
Start the local FastAPI server and Web UI playground:
```bash
cd graphrag_agent
uv run adk web
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to chat with your papers.

---

## Alternative LLM Providers (LM Studio / llama.cpp)

Instead of Ollama, the agent can be configured to route queries to any local provider serving an OpenAI-compatible API (such as LM Studio or a llama.cpp server).

To configure an alternative local runner:
1. Ensure your model prefix in `docker-compose.yml` (or environment variables) starts with `openai/` (e.g. `openai/gemma-2-9b-it` or `openai/llama.cpp`).
2. Point `LLM_API_BASE` to your provider's endpoint:
   * **LM Studio**: `http://host.docker.internal:1234/v1`
   * **llama.cpp**: `http://host.docker.internal:8080/v1` (or using the containerized `http://llama-cpp:8080/v1` service by uncommenting `llama-cpp` in `docker-compose.yml`).
3. Set `AGENT_LLM_MODEL`, `RAG_LLM_MODEL`, and `RAG_EMBEDDER_MODEL` to match your loaded models.

---

## Docker Compose Deployment
To deploy the entire system (Agent + FalkorDB + Ollama/llama.cpp) on any target machine:

1. Clone or copy `docker-compose.yml`, `DEPLOY.md`, and the `graphrag_agent/` directory to the target machine.
2. Spin up the container stack:
   ```bash
   docker compose up -d --build
   ```
3. Initialize the models inside the Ollama container (if using Ollama):
   ```bash
   docker compose exec ollama ollama pull gemma4:latest
   docker compose exec ollama ollama pull nomic-embed-text:latest
   ```
4. Access the playground at **[http://localhost:8000](http://localhost:8000)**.

*For detailed instructions on mounting custom GGUF models or configuring NVIDIA GPU drivers, read the **[DEPLOY.md](file:///home/fony/graphrag-falkordb/DEPLOY.md)** file.*
