# 🧠 NeuralTwin: Codebase Context & Architecture Overview

> **Note to AI Assistants:** This document provides the full context, directory structure, and technical stack of the NeuralTwin project. Use this to understand the system architecture when answering questions or generating code.

## 1. Project Identity
*   **Name:** NeuralTwin
*   **Type:** Production-Grade Agentic RAG System
*   **Base:** Enhanced fork of "LLM Engineer's Handbook" (Packt)
*   **Key Differentiators:** GraphRAG (Neo4j), Agentic Workflow (ReAct), HyDE (Hypothetical Document Embeddings), Production MLOps (ZenML, Kafka).

## 2. Directory Structure & Key Files

```
neuraltwin/
├── .github/                # CI/CD Workflows (GitHub Actions)
├── configs/                # Hydra configurations (ETL, Training, Inference)
├── docs/                   # Documentation (Architecture, Guides)
├── k8s/                    # Kubernetes Manifests (Deployments, Services)
├── llm_engineering/        # SOURCE CODE (DDD Architecture)
│   ├── application/        # Application Layer (Use Cases)
│   │   ├── agents/         # ReAct Agents (ResearchAgent)
│   │   ├── crawlers/       # Web Scrapers (Github, Medium, LinkedIn)
│   │   ├── graph/          # Graph Ingestion Logic (Neo4j)
│   │   ├── rag/            # RAG Pipeline (Retriever, Reranker, HyDE)
│   │   └── utils/          # Helpers
│   ├── domain/             # Domain Layer (Business Logic)
│   │   ├── base/           # Base Classes (NoSQLBaseDocument)
│   │   ├── chunks/         # Chunking Logic
│   │   └── documents/      # Data Models (User, Document, Repository)
│   ├── infrastructure/     # Infrastructure Layer (External Interfaces)
│   │   ├── aws/            # AWS Clients (S3, SageMaker)
│   │   ├── db/             # Database Clients (Mongo, Qdrant)
│   │   ├── graph/          # Graph DB Clients (Neo4jAdapter)
│   │   └── inference_pipeline_api.py # FastAPI Entrypoint
│   └── model/              # Model Layer (Evaluation, Fine-tuning)
├── pipelines/              # ZenML Pipeline Definitions
├── steps/                  # ZenML Pipeline Steps
├── tests/                  # Pytest Suite (Unit, Integration, Load)
├── tools/                  # Utility Scripts (Graph Ingestion, Agent Demo)
├── docker-compose.yml      # Local Services (Mongo, Qdrant, Neo4j, Zookeeper, Kafka)
└── Makefile               # Task Runner (ETL, Start, Stop, Test)
```

## 3. Technology Stack

### Core
*   **Language:** Python 3.11
*   **Framework:** FastAPI (API), ZenML (MLOps), Pydantic (Validation)

### Data & Storage
*   **Vector DB:** Qdrant (Semantic Search)
*   **Document DB:** MongoDB (Metadata & Raw Content)
*   **Graph DB:** Neo4j (Entity Relationships)
*   **Cache:** Redis (Semantic Caching, Rate Limiting)
*   **Queue:** Apache Kafka (Event-Driven Data Ingestion)

### AI & Retrieval
*   **LLM:** Llama 3.1 8B (Served via Ollama, vLLM, or VLLM)
*   **Inference Engine:** vLLM (Production), Ollama (Local/Dev)
*   **Embeddings:** OpenAI `text-embedding-3-small` / `all-MiniLM-L6-v2`
*   **Reranker:** `cross-encoder/ms-marco-MiniLM-L-12-v2`
*   **Agent Framework:** Custom ReAct Loop (No LangChain bloat)

### Operations
*   **Containerization:** Docker & Docker Compose
*   **Orchestration:** Kubernetes (K8s)
*   **Observability:** Prometheus, Grafana, Jaeger, Opik, Comet ML

## 4. Key Workflows / Flows

### A. Data Ingestion (ETL)
`Crawler` -> `Kafka (raw_data)` -> `Cleaner/Chunker` -> `Embedder` -> `Kafka (embeddings)` -> `Qdrant` + `MongoDB` + `Neo4j`

### B. RAG Inference (Hybrid + Graph)
1.  **User Query** -> **HyDE Generator** (Hypothetical Answer)
2.  **Parallel Search:**
    *   **Dense:** Vector Search in Qdrant (using HyDE vector)
    *   **Sparse:** BM25 Keyword Search
    *   **Graph:** Cypher Query in Neo4j (2-hop traversal)
3.  **Fusion:** RRF (Reciprocal Rank Fusion) combines logic.
4.  **Reranking:** Cross-Encoder filters top results.
5.  **Generation:** LLM synthesizes answer with citations.

### C. Agentic Workflow
1.  **Agent** receives complex query.
2.  **Thought:** "I need to check relations between services."
3.  **Action:** Calls `search_graph` tool.
4.  **Observation:** Receives graph data.
5.  **Thought:** "Now I need code examples."
6.  **Action:** Calls `search_vector` tool.
7.  **Synthesis:** Combines structured and unstructured data.

## 5. Current State & Implementation Details
*   **Agentic Status:** The `ResearchAgent` is fully implemented in code (`llm_engineering/application/agents/research_agent.py`) but operates in "Mock Mode" by default for demo purposes (to avoid API costs during development). It is architecturally ready for real LLM switching.
*   **GraphRAG:** Fully functional ingestion script (`tools/run_graph_ingestion.py`) and retrieval adapter (`Neo4jAdapter`).
