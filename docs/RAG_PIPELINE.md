# RAG Pipeline Deep Dive

## 🧠 Retreival Strategy

The system uses a **Hybrid Retrieval** approach to maximize both precision and recall.

### 1. Dense Retrieval (Semantic Search)
-   **Model:** `sentence-transformers/all-MiniLM-L6-v2` (or OpenAI `text-embedding-3-small` in prod).
-   **Mechanism:** Cosine similarity between query and document embeddings.
-   **Advantage:** Captures semantic meaning (e.g., "coding" matches "programming").

### 2. Sparse Retrieval (Keyword Search)
-   **Model:** BM25 (Best Match 25).
-   **Mechanism:** Token overlap and TF-IDF weighting.
-   **Advantage:** Captures exact keyword matches and domain-specific jargon that embeddings might miss.

### 3. Reciprocal Rank Fusion (RRF)
We combine results using RRF:
$$RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$
Where $r(d)$ is the rank of document $d$ in the results $R$, and $k$ is a constant (typically 60). This ensures no single retriever dominates the results.

## 🎯 Reranking

After retrieving the top candidate documents (e.g., 50), we use a **Cross-Encoder** to re-score them.

-   **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
-   **Process:** The model takes pairs of `(Query, Document)` and outputs a relevance score [0, 1].
-   **Result:** The Top-5 highest scored documents are passed to the LLM context.

## ⚡ Performance Optimizations

1.  **Semantic Caching:**
    -   We cache the *query embedding* and its search results.
    -   New queries near-identical to cached ones (cosine similarity > 0.95) return cached results instantly.

2.  **Context Optimization:**
    -   Prompt structure is tuned to minimize token usage while maximizing instruction following.
    -   Context chunks are ordered by relevance to mitigate the "Lost in the Middle" phenomenon.

## 📊 Evaluation

We evaluate the pipeline using `rag_metrics.py` (see `evaluation/` folder):
-   **Precision@K:** Percentage of relevant docs in top K.
-   **MRR:** Rank of the first relevant document.
-   **Latency:** End-to-end response time.
