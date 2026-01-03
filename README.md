# NeuralTwin: Production-Grade RAG System
> **An End-to-End LLM System demonstrating Advanced RAG, MLOps, and Efficient Fine-Tuning.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![ZenML](https://img.shields.io/badge/ZenML-0.56.1-purple)](https://zenml.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Project Overview

**NeuralTwin** is a portfolio project designed to showcase **Production-Level Machine Learning Engineering**. Unlike simple RAG tutorials, this system implements a complete lifecycle: from data ingestion to inference, wrapped in robust MLOps practices.

It acts as a "Second Brain" for software engineers, ingesting technical content (GitHub, Medium, LinkedIn) and allowing users to query it using state-of-the-art Retrieval Augmented Generation (RAG) techniques.

**Key Differentiators:**
*   **Production-Ready:** Dockerized, CI/CD ready, with comprehensive logging and monitoring.
*   **Advanced RAG:** Implements **Hybrid Search (Dense + Sparse)** and **Cross-Encoder Reranking**.
*   **Cost-Efficient:** Optimized to run on consumer hardware (T4 GPU) or completely free (Mock Mode).

---

## 🏗️ System Architecture

The system follows a modular microservices architecture, orchestrated by ZenML.

```mermaid
graph TD
    User[User / Client] -->|HTTP Request| API[FastAPI Inference Service]
    API -->|Log Traces| Opik[Opik Observability]
    
    subgraph "Inference Flow (RAG)"
        API -->|1. Query| Retriever[Context Retriever]
        Retriever -->|2. Hybrid Search| VectorDB[(Qdrant Vector DB)]
        VectorDB -->|3. Raw Candidates| Reranker[Cross-Encoder Reranker]
        Reranker -->|4. Top Context| LLM[LLM Service (Llama 3 / GPT-4)]
        LLM -->|5. Response| API
    end

    subgraph "Data Pipeline (ETL)"
        Sources[GitHub / Medium / LinkedIn] -->|Crawl| Crawler[Scrapers]
        Crawler -->|Raw Data| Mongo[(MongoDB)]
        Mongo -->|Process| ETL[ZenML ETL Pipeline]
        ETL -->|Embed & Index| VectorDB
    end
```

---

## 🚀 Key Features

### 1. Advanced RAG Implementation
*   **Hybrid Retrieval:** Combines semantic search (embeddings) with keyword search (BM25) using **Reciprocal Rank Fusion (RRF)** for optimal recall.
*   **Reranking:** Utilizes a Cross-Encoder (`ms-marco-MiniLM`) to precision-score retrieved documents.
*   **Streaming API:** Server-Sent Events (SSE) for real-time token streaming, improving perceived latency.

### 2. MLOps & Infrastructure
*   **Orchestration:** **ZenML** manages reproducible pipelines for ETL and Training.
*   **Experiment Tracking:** **Comet ML** tracks hyperparameters and training metrics.
*   **Observability:** **Opik** provides trace-level visibility into LLM chains and prompt costs.
*   **Containerization:** Fully Dockerized Setup with `docker-compose` and health checks.

### 3. Efficient Fine-Tuning (Showcase)
*   **QLoRA:** Implements 4-bit Quantized Low-Rank Adaptation for fine-tuning **Llama 3 8B** on a single T4 GPU (16GB VRAM).
*   **Flash Attention:** Optimized attention kernels for faster training throughput.

---

## 🛠️ Usage Guide

### 1. Prerequisites
*   Docker & Docker Compose
*   Make (Optional, for easy commands)

### 2. Automatic Setup
We provide a unified setup script to configure your environment and start the infrastructure.

```bash
# 1. Clone the repository
git clone https://github.com/ductaip/LLM-Engineers-Handbook-main
cd LLM-Engineers-Handbook-main

# 2. Setup Environment & Dependencies
make setup

# 3. Start Infrastructure (MongoDB + Qdrant)
make start
```

### 3. Running the System

**Option A: Crawl Data (ETL)**
Ingest your GitHub Repos and Articles (Configured in `configs/digital_data_etl_paul_iusztin.yaml`).
```bash
make etl
```

**Option B: Start RAG Server**
Launch the inference API locally.
```bash
make rag-server
```

**Option C: Test Query**
Send a request to your local RAG twin.
```bash
make rag-test
```

---

## 📂 Project Structure

```bash
.
├── configs/                 # YAML Configuration files for pipelines
├── docs/                    # Detailed architectural documentation
├── evaluation/              # RAG metric evaluation tools
├── llm_engineering/         # Core application code
│   ├── application/         # Business logic (Crawlers, RAG, Reranking)
│   ├── domain/              # Data models and exceptions
│   └── infrastructure/      # Database connectors & API endpoints
├── pipelines/               # ZenML Pipeline definitions
├── steps/                   # Individual pipeline steps (clean, chunk, embed)
├── tools/                   # CLI tools for running pipelines
├── training/                # Fine-tuning scripts (QLoRA/Peft)
├── Makefile                 # Shortcuts for common commands
└── pyproject.toml           # Dependency management
```

---

## 🔌 Configuration

Refine your personal data sources in `configs/digital_data_etl_paul_iusztin.yaml`.

```yaml
parameters:
  user_full_name: Phan Duc Tai
  links:
    - https://github.com/ductaip/Notebook-Algorithm
    - https://github.com/ductaip/NestJS-Ecommerce
```

For **API Keys** (OpenAI, Comet, etc.), create a `.env` file (automated by `make setup`).
The system supports a **Mock Mode** (`MOCK_LLM=true`) to run completely free of charge.

---

## 👨‍💻 Author

**Phan Duc Tai**

*   **GitHub:** [github.com/ductaip](https://github.com/ductaip)
*   **LinkedIn:** [linkedin.com/in/phanductai](https://www.linkedin.com/in/phanductai/)

This project is an evolution of the *LLM Engineer's Handbook*, refactored and personalized to demonstrate advanced capability in building scalable AI systems.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
