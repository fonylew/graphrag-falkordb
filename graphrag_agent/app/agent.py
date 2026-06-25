# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
import os
import google.auth

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM as RAGLiteLLM, LiteLLMEmbedder

# Set environment variables for Ollama and local execution
if "OLLAMA_API_BASE" not in os.environ:
    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
if "GOOGLE_GENAI_USE_VERTEXAI" not in os.environ:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# Try getting project_id, but fallback gracefully if not authenticated
try:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        os.environ["GOOGLE_CLOUD_PROJECT"] = "local-project"

async def query_graphrag(question: str) -> str:
    """Queries the GraphRAG knowledge graph to answer questions about the research paper
    'Realizing Memory-Optimized Distributed Graph Processing' and related topics.
    
    Use this tool whenever the user asks about the research paper, distributed graph processing,
    memory optimization in graphs, metadata, authors (like K. Liao, D. Logothetis, M. Kabiljo),
    or any other entities/relationships in the database.

    Args:
        question: A string containing the question or search query to query the GraphRAG with.

    Returns:
        The synthesized answer with relevant facts retrieved from the graph database.
    """
    ollama_api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    falkordb_host = os.environ.get("FALKORDB_HOST", "localhost")
    falkordb_port = int(os.environ.get("FALKORDB_PORT", "6379"))

    rag_llm_model = os.environ.get("RAG_LLM_MODEL", "ollama/gemma4:latest")
    rag_embedder_model = os.environ.get("RAG_EMBEDDER_MODEL", "ollama/nomic-embed-text:latest")
    llm_api_base = os.environ.get("LLM_API_BASE", ollama_api_base)

    if rag_llm_model.startswith("openai/"):
        os.environ["OPENAI_API_BASE"] = llm_api_base
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "local"

    llm = RAGLiteLLM(
        model=rag_llm_model,
        api_base=llm_api_base,
    )
    embedder = LiteLLMEmbedder(
        model=rag_embedder_model,
        api_base=llm_api_base,
    )
    
    rag = GraphRAG(
        connection=ConnectionConfig(
            host=falkordb_host,
            port=falkordb_port,
            graph_name="fonylew/GraphRAG",
        ),
        llm=llm,
        embedder=embedder,
        embedding_dimension=768,
    )
    
    try:
        async with rag:
            response = await rag.completion(question, return_context=False)
            return response.answer
    except Exception as e:
        return f"Error querying GraphRAG: {str(e)}"


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        query: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


# Define the agent model using environment overrides
agent_llm_model = os.environ.get("AGENT_LLM_MODEL", "ollama_chat/gemma4:latest")
agent_api_base = os.environ.get("LLM_API_BASE", "http://localhost:11434")

if agent_llm_model.startswith("openai/"):
    os.environ["OPENAI_API_BASE"] = agent_api_base
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "local"

root_agent = Agent(
    name="root_agent",
    model=LiteLlm(
        model=agent_llm_model,
    ),
    instruction=(
        "You are a helpful AI research assistant. You have access to a GraphRAG "
        "knowledge graph containing a database of research papers. Use the query_graphrag "
        "tool to retrieve accurate facts and answer any questions about the papers or "
        "graph database contents."
    ),
    tools=[query_graphrag, get_current_time],
)

app = App(
    root_agent=root_agent,
    name="graphrag_agent",
)
