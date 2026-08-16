"""BM25 sparse retrieval over identifier-tokenized code chunks.

In-memory `rank_bm25.BM25Okapi`, built by scrolling the chunks already in Qdrant for a
repo rather than a second index in a separate store. Qdrant supports named sparse
vectors for a "real" production setup, but that means re-plumbing collection config and
the upsert path for every future language; an in-memory index rebuilt from data already
written is a smaller, still-real, still-unmocked way to get BM25 working now. Revisit if
corpus size makes the rebuild cost matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

from codeatlas.application.rag.identifier_tokenizer import tokenize
from codeatlas.application.rag.models import RetrievedChunk
from codeatlas.domain.code.chunk import CodeChunkDocument
from codeatlas.infrastructure.db.qdrant import connection

_SCROLL_PAGE_SIZE = 512


@dataclass
class _BM25Index:
    bm25: BM25Okapi
    payloads: list[dict] = field(default_factory=list)


class SparseCodeRetriever:
    _index_cache: ClassVar[dict[str, _BM25Index]] = {}

    def __init__(self, repo_id: str):
        self.repo_id = repo_id

    def invalidate_cache(self) -> None:
        self._index_cache.pop(self.repo_id, None)

    def _build_index(self) -> _BM25Index:
        payloads: list[dict] = []
        offset = None
        query_filter = Filter(
            must=[FieldCondition(key="repo_id", match=MatchValue(value=self.repo_id))]
        )
        while True:
            records, offset = connection.scroll(
                collection_name=CodeChunkDocument.get_collection_name(),
                scroll_filter=query_filter,
                limit=_SCROLL_PAGE_SIZE,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            payloads.extend(r.payload for r in records)
            if offset is None:
                break

        tokenized_corpus = [tokenize(p["content"]) for p in payloads]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else BM25Okapi([[]])
        return _BM25Index(bm25=bm25, payloads=payloads)

    def _get_index(self) -> _BM25Index:
        if self.repo_id not in self._index_cache:
            self._index_cache[self.repo_id] = self._build_index()
        return self._index_cache[self.repo_id]

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        index = self._get_index()
        if not index.payloads:
            return []

        scores = index.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[RetrievedChunk] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                break
            payload = index.payloads[i]
            results.append(
                RetrievedChunk(
                    content=payload["content"],
                    qualified_name=payload["qualified_name"],
                    file_path=payload["file_path"],
                    start_line=payload["start_line"],
                    end_line=payload["end_line"],
                    source="sparse",
                    score=float(scores[i]),
                    docstring=payload.get("docstring"),
                    signature=payload.get("signature"),
                )
            )
        return results
