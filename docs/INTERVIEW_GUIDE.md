# NeuralTwin System Design & Interview Guide

This guide is designed to help you explain the architectural decisions, trade-offs, and scalability strategies of the NeuralTwin project during a technical interview.

## 🏗️ Core Architectural Decisions

### 1. Why Event-Driven Architecture (Kafka)?

**Q: Why did you introduce Kafka instead of just calling services synchronously?**

**A:**
*   **Decoupling:** Kafka decouples the Data Ingestion (Producers) from the Embedding/Indexing (Consumers). This allows the crawler to run at its own speed without overwhelming the embedding model or the vector database.
*   **Backpressure Handling:** during high-load crawling sessions, Kafka acts as a buffer. If the embedding service is slow or down, no data is lost; it just accumulates in the topic.
*   **Scalability:** We can easily scale the `DocumentProcessorConsumer` horizontally by adding more consumer instances to the same consumer group. Kafka automatically rebalances partitions among them.

### 2. Why Agentic RAG?

**Q: Why use an Agent (ReAct) instead of a linear RAG chain?**

**A:**
*   **Dynamic Decision Making:** A linear chain always performs the same steps (Retrieve -> Generate). An Agent can *decide* what to do. For example, if the retrieved context is insufficient, the Agent can choose to perform a web search or ask a clarifying question.
*   **Tool Use:** The Agent can use tools like `Calculator` or `WebSearch`. This creates a much more robust system that can handle complex queries requiring multi-step reasoning, not just retrieval.

### 3. Database Choices

**Q: Why MongoDB + Qdrant? Why not just one?**

**A:**
*   **MongoDB (The "Source of Truth"):** Stores the raw, unstructured JSON data from crawlers. It's flexible and allows us to store metadata alongside the content. It serves as the "Cold Node" storage.
*   **Qdrant (The "Hot Node"):** Optimized purely for vector similarity search. Storing the full document text in the vector DB can be expensive and slow. We store vectors and minimal payload in Qdrant, and if we need the full original context, we can fetch it from Mongo (though often the chunk text is sufficient).
*   **Polyglot Persistence:** Using the right tool for the job. Relational DBs (Postgres) would be too rigid for scraped schema-less data.

### 4. Why GraphRAG?

**Q: Vector Databases are popular. Why did you add a Graph Database (Neo4j)?**

**A:**
*   **LIMITATION of Vectors:** Vector search is great for *similarity* (finding things that look like X), but bad at *structure* or *relationships*. It cannot easily answer "How does module A relate to module B?" if they don't share similar words.
*   **The "Graph" Advantage:** Neo4j stores entities and their relationships (e.g., `(JWT)-[:USED_FOR]->(Authentication)`). This allows the Agent to perform "multi-hop reasoning"—following the edges of the graph to understand complex dependencies that pure vector search would miss.

### 5. Why HyDE?

**Q: What is HyDE and why use it?**

**A:**
*   **The Problem:** Users ask short, vague questions (e.g., "How auth works?"). Documents are long and detailed. The vector overlap between the query and the document is often poor.
*   **The Solution:** We use an LLM to hallucinate a "perfect" hypothetical answer. This hypothetical answer contains the *vocabulary* and *structure* of the target document. We embed *that* instead of the query. This bridges the semantic gap and significantly boosts recall.

### 6. Why use the Facade Pattern?

**Q: Why introduce a Facade (`NeuralTwinAIFacade`) between the API and the application layer?**

**A:**
*   **Encapsulation:** The Facade hides the complex orchestration of multiple AI services (retrievers, HyDE generators, reasoning extractors) behind a single, clean interface.
*   **Loose Coupling:** FastAPI routers import only the Facade, remaining completely agnostic of slow-loading ML libraries (LangChain, Qdrant, vLLM). This keeps the transport layer lightweight.
*   **Lazy Initialization:** Heavy ML components are created on first use, preventing massive memory allocations and slow server start-ups.
*   **Testability:** In unit tests, we can easily swap the entire Facade with a mock that returns pre-canned responses, enabling fast and deterministic API testing.

## 🚀 Scaling & Production Readiness

### 1. How do you scale this system?

*   **Stateless API:** The FastAPI layer is stateless. We can spin up N replicas behind a Load Balancer (K8s Service/Ingress).
*   **Async Processing:** Heavy lifting (PDF parsing, Embedding) is offloaded to Kafka consumers, which can be scaled independently of the API.
*   **Database Sharding:**
    *   **Qdrant:** Supports distributed deployment and sharding.
    *   **MongoDB:** Supports sharding and replica sets for high availability.
*   **Caching:** Redis handles 70-80% of repeat queries, keeping load off the expensive Vector DB and LLM.

### 2. How do you handle Security?

*   **Authentication:** JWT (JSON Web Tokens) for stateless auth. No session lookup needed on DB.
*   **Rate Limiting:** Redis-based sliding window limiter. Prevents DDoS and API abuse (Token exhaustion attacks).
*   **Secret Management:** In K8s, we use Kubernetes Secrets (or Vault) instead of `.env` files.

### 3. Observability

*   **Prometheus:** Tracks technical metrics (Latencies, Throughput, Error rates, Memory/CPU).
*   **Grafana:** Visualizes these metrics for SRE dashboarding.
*   **Opik:** Tracks *LLM specific* metrics (Token usage, hallucination scores, trace chains).
*   **Jaeger:** Distributed tracing to see exactly where a request spent time (API -> Redis -> Qdrant -> LLM).

## 💡 Potential Improvements (Self-Correction)

*   **Canary Deployments:** Implementing Flagger (Argo Rollouts) to gradually shift traffic to new model versions.
*   **Fine-Grained Access Control:** Implementing Row-Level Security (RLS) in Postgres/Supabase for multi-tenant data isolation.
