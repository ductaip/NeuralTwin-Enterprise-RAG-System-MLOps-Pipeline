from typing import Protocol, TypeVar

import opik

from codeatlas.application.networks import CrossEncoderModelSingleton
from codeatlas.domain.queries import Query

from .base import RAGStep


class _HasContent(Protocol):
    content: str


ChunkT = TypeVar("ChunkT", bound=_HasContent)


class Reranker(RAGStep):
    """Cross-encoder reranker. Generic over any chunk type with a `.content` field —
    used with `RetrievedChunk` (code retrieval) but doesn't otherwise care what domain
    the chunk came from."""

    def __init__(self) -> None:
        self._model = CrossEncoderModelSingleton()

    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[ChunkT], keep_top_k: int) -> list[ChunkT]:
        query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        scores = self._model(query_doc_tuples)

        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents
