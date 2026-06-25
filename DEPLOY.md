# GraphRAG System Deployment Guide

This guide details how to deploy the FalkorDB GraphRAG system (Ollama, FalkorDB, and ADK Web Agent) using Docker and Docker Compose.

## Prerequisites
1. **Docker & Docker Compose** installed on the destination machine.
2. *(Optional)* **NVIDIA Container Toolkit** installed if you want to run Ollama with GPU acceleration on an NVIDIA GPU.

---

## Step 1: Clone / Copy Files
Copy the following files to the destination machine:
- `docker-compose.yml`
- `graphrag_agent/` (including `Dockerfile`, `pyproject.toml`, `uv.lock`, and the `app/` folder)

---

## Step 2: Configure GPU Support (Optional)
If the destination machine has an NVIDIA GPU and NVIDIA Container Toolkit installed:
1. Open `docker-compose.yml`.
2. Uncomment the `deploy` configuration block under the `ollama` service:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

---

## Step 3: Start the Services
Run the following command to start all services in the background:
```bash
docker compose up -d --build
```
This will:
1. Build the local `graphrag_agent` Docker image using Python 3.11.
2. Spin up FalkorDB (on port `6379`), Ollama (on port `11434`), and the Agent (on port `8000`).

---

## Step 4: Pull Local Models to Ollama
Once the Ollama container is running, pull the required models:
```bash
# Pull the LLM model
docker compose exec ollama ollama pull gemma4:latest

# Pull the Embedding model
docker compose exec ollama ollama pull nomic-embed-text:latest
```

---

## Step 5: Ingest PDF Documents into FalkorDB
Since FalkorDB and Ollama port-forward to `localhost` on the host machine, you can run data ingestion directly from your host if you have Python installed:
```bash
# Sync host dependencies
uv sync

# Ingest your PDF papers (e.g., Zotero PDFs)
uv run python manage_rag.py ingest --path /path/to/papers/
```
Alternatively, if you want to clear or check database status:
```bash
# Check database ingestion status
uv run python manage_rag.py status

# Clear the database
uv run python manage_rag.py clear
```

---

## Step 6: Use the Agent Web Playground
Access the ADK Web UI playground at:
👉 **[http://localhost:8000](http://localhost:8000)**

You can ask the agent questions, and it will use the `query_graphrag` tool to answer them using facts retrieved from your local FalkorDB GraphRAG database.

---

## Alternative LLM Providers (LM Studio / llama.cpp)

If you prefer to serve your local models using **LM Studio (LMS)** or **llama.cpp** instead of Ollama, both serve OpenAI-compatible endpoints that can be integrated using the environment variables in `docker-compose.yml`.

### How it Works
1. Since the API is OpenAI-compatible, model names must be prefixed with `openai/` (e.g. `openai/gemma-2-9b-it` or `openai/llama.cpp`).
2. Provide `LLM_API_BASE` pointing to the provider's server.
3. If the server is running on the host machine (outside the Docker network), you can use `http://host.docker.internal:<port>/v1` to route to it.

### Step 1: Allow Container-to-Host Routing
To let your Docker containers access services running directly on the host machine (e.g., LM Studio running on your desktop), add the `extra_hosts` block to the `agent` service in your `docker-compose.yml`:
```yaml
  agent:
    ...
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### Step 2: Configure Environment Variables

Update the `environment:` section of the `agent` service:

#### Option A: Using LM Studio (Default Port `1234`)
```yaml
    environment:
      - FALKORDB_HOST=falkordb
      - FALKORDB_PORT=6379
      - SESSION_SERVICE_URI=sqlite:////code/.adk/session.db
      - GOOGLE_GENAI_USE_VERTEXAI=False
      - GOOGLE_CLOUD_PROJECT=local-project
      - OTEL_TO_CLOUD=False
      
      # LM Studio Config
      - AGENT_LLM_MODEL=openai/gemma-2-9b-it           # Match the loaded model ID in LMS
      - RAG_LLM_MODEL=openai/gemma-2-9b-it
      - RAG_EMBEDDER_MODEL=openai/text-embedding-nomic  # If embedding locally via LMS
      - LLM_API_BASE=http://host.docker.internal:1234/v1
```

#### Option B: Using llama.cpp Server (Default Port `8080`)

You can run `llama.cpp` either on your host system or containerized inside your Docker network by uncommenting the `llama-cpp` service in your `docker-compose.yml`:

1. Create a local `./models/` directory on the destination machine and place your `.gguf` model file inside it.
2. In `docker-compose.yml`, uncomment the `llama-cpp` service block and specify your model filename in the `command:` line.
3. Configure the `environment:` section of the `agent` service:
```yaml
    environment:
      - FALKORDB_HOST=falkordb
      - FALKORDB_PORT=6379
      - SESSION_SERVICE_URI=sqlite:////code/.adk/session.db
      - GOOGLE_GENAI_USE_VERTEXAI=False
      - GOOGLE_CLOUD_PROJECT=local-project
      - OTEL_TO_CLOUD=False
      
      # llama.cpp Config (containerized)
      - AGENT_LLM_MODEL=openai/llama.cpp                # Value is ignored by llama.cpp but required by SDK
      - RAG_LLM_MODEL=openai/llama.cpp
      - RAG_EMBEDDER_MODEL=openai/llama.cpp
      - LLM_API_BASE=http://llama-cpp:8080/v1            # Routes to the containerized llama.cpp service
```
*(Note: If you run llama.cpp server on the host machine directly rather than containerized, set `LLM_API_BASE=http://host.docker.internal:8080/v1` instead).*

Once configured, rebuild the agent container with `docker compose up -d --build`. The agent will seamlessly route all inference queries and GraphRAG completions to your custom server.

