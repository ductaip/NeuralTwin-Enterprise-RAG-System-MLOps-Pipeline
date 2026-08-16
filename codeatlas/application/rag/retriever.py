"""Code retrieval: fan out dense/sparse/graph, fuse with RRF, rerank with cross-encoder.

Replaces the NeuralTwin-era `ContextRetriever`, which searched `EmbeddedPostChunk` /
`EmbeddedArticleChunk` / `EmbeddedRepositoryChunk` — blog-post domain models with no
equivalent in CodeAtlas. Same class name (existing callers in `ai_facade.py`,
`inference_pipeline_api.py`, `tools/rag.py` construct `ContextRetriever(...)`), new
domain: a `CodeChunkDocument` collection scoped by `repo_id`, per spec §2.1 "domain/code/".
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field

import opik
from loguru import logger

from codeatlas.application.rag.dense_retriever import DenseCodeRetriever
from codeatlas.application.rag.graph_retriever import GraphRetriever
from codeatlas.application.rag.models import RetrievedChunk
from codeatlas.application.rag.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from codeatlas.application.rag.sparse_retriever import SparseCodeRetriever
from codeatlas.domain.queries import Query

from .hyde_generator import HydeGenerator
from .reranking import Reranker


@dataclass
class RetrievalTrace:
    """Per-retriever + fused + final results, for the Phase 2 verification step
    ("in ra top-5 của TỪNG retriever riêng lẻ ... và sau khi fuse+rerank")."""

    dense: list[RetrievedChunk] = field(default_factory=list)
    sparse: list[RetrievedChunk] = field(default_factory=list)
    graph: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    final: list[RetrievedChunk] = field(default_factory=list)
    hypothetical_code: str = ""


class ContextRetriever:
    def __init__(self, repo_id: str, fetch_k: int = 15, rrf_k: int = DEFAULT_RRF_K):
        self.repo_id = repo_id
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

        self._dense = DenseCodeRetriever(repo_id)
        self._sparse = SparseCodeRetriever(repo_id)
        self._graph = GraphRetriever(repo_id)
        self._hyde_generator = HydeGenerator()
        self._reranker = Reranker()

    @opik.track(name="ContextRetriever.search")
    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return self.search_with_trace(query, k).final

    def search_with_trace(self, query: str, k: int = 5) -> RetrievalTrace:
        hypothetical_code = self._hyde_generator.generate(query)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            dense_future = executor.submit(self._dense.search, hypothetical_code, self.fetch_k)
            sparse_future = executor.submit(self._sparse.search, query, self.fetch_k)
            graph_future = executor.submit(self._graph.search, query, self.fetch_k)

            dense = dense_future.result()
            sparse = sparse_future.result()
            graph = graph_future.result()

        logger.info(
            f"Retrieved dense={len(dense)} sparse={len(sparse)} graph={len(graph)} "
            f"for query: {query[:80]}"
        )

        fused = reciprocal_rank_fusion([dense, sparse, graph], k=self.rrf_k)
        final = self.rerank(query, fused, keep_top_k=k) if fused else []

        return RetrievalTrace(
            dense=dense, sparse=sparse, graph=graph, fused=fused, final=final,
            hypothetical_code=hypothetical_code,
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk], keep_top_k: int) -> list[RetrievedChunk]:
        query_model = Query.from_str(query)
        reranked = self._reranker.generate(query=query_model, chunks=chunks, keep_top_k=keep_top_k)
        logger.info(f"{len(reranked)} chunks reranked successfully.")
        return reranked
