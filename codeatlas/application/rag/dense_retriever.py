from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchValue

from codeatlas.application.networks.embeddings import EmbeddingModelSingleton
from codeatlas.application.rag.models import RetrievedChunk
from codeatlas.domain.code.chunk import CodeChunkDocument
from codeatlas.infrastructure.db.qdrant import connection


class DenseCodeRetriever:
    """Cosine similarity over `CodeChunkDocument` embeddings, scoped to one repo."""

    def __init__(self, repo_id: str, embedder: EmbeddingModelSingleton | None = None):
        self.repo_id = repo_id
        self._embedder = embedder or EmbeddingModelSingleton()

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        vector = self._embedder(query, to_list=True)
        records = connection.search(
            collection_name=CodeChunkDocument.get_collection_name(),
            query_vector=vector,
            query_filter=Filter(
                must=[FieldCondition(key="repo_id", match=MatchValue(value=self.repo_id))]
            ),
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            RetrievedChunk(
                content=r.payload["content"],
                qualified_name=r.payload["qualified_name"],
                file_path=r.payload["file_path"],
                start_line=r.payload["start_line"],
                end_line=r.payload["end_line"],
                source="dense",
                score=r.score,
                docstring=r.payload.get("docstring"),
                signature=r.payload.get("signature"),
            )
            for r in records
        ]
