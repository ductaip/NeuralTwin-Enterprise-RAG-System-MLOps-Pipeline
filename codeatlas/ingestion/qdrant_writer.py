"""Embed chunks produced by `chunker.py` and upsert them into Qdrant.

This is the piece Phase 1 built but never wired up: `chunk_module` existed, but nothing
called it from the ingestion CLI, so no code chunk ever reached the vector store. Kept
separate from `graph_builder.py` because the two write to different databases and either
one should be able to fail without corrupting the other.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from loguru import logger

from codeatlas.domain.code.chunk import CodeChunkDocument
from codeatlas.ingestion.chunker import chunk_module
from codeatlas.ingestion.models import CodeChunk, ParsedModule, SourceFile, SymbolDef

if TYPE_CHECKING:
    from codeatlas.application.rag.contextual_enrichment import ContextualEnricher

Embedder = Callable[[list[str]], list[list[float]]]

_QDRANT_ID_NAMESPACE = uuid.UUID("5b1f2e7a-8c3d-4b2a-9e1f-6a7d8c9b0e1f")


def deterministic_chunk_id(repo_id: str, qualified_name: str, part: int) -> uuid.UUID:
    """Point ID derived from (repo_id, qualified_name, part) rather than random.

    `CodeChunkDocument.id` defaults to `uuid.uuid4()` — fine for a single write, but it
    means re-ingesting the same commit duplicates every chunk instead of overwriting it,
    unlike `graph_builder.py`'s MERGE-on-key idempotency. Qdrant upsert-by-id gives the
    same guarantee once the id is deterministic.
    """
    return uuid.uuid5(_QDRANT_ID_NAMESPACE, f"{repo_id}:{qualified_name}:{part}")


@dataclass
class QdrantWriteStats:
    chunks_written: int = 0


class QdrantChunkWriter:
    def __init__(
        self,
        embedder: Embedder | None = None,
        batch_size: int = 64,
        enricher: "ContextualEnricher | None" = None,
    ):
        self._embedder = embedder or self._default_embedder()
        self._batch_size = batch_size
        self._enricher = enricher

    @staticmethod
    def _default_embedder() -> Embedder:
        from codeatlas.application.networks.embeddings import EmbeddingModelSingleton

        model = EmbeddingModelSingleton()
        return lambda texts: model(texts, to_list=True)

    def write(
        self,
        repo_id: str,
        commit_sha: str | None,
        language: str,
        modules: Sequence[tuple[SourceFile, ParsedModule]],
        symbols_by_qn: dict[str, SymbolDef],
    ) -> QdrantWriteStats:
        CodeChunkDocument.get_or_create_collection()

        all_chunks: list[CodeChunk] = [
            chunk
            for source, parsed in modules
            for chunk in chunk_module(parsed, source, repo_id=repo_id, language=language)
        ]

        context_by_qn: dict[str, str] = {}
        if self._enricher is not None and all_chunks:
            context_by_qn = self._enrich_all(repo_id, all_chunks)

        stats = QdrantWriteStats()
        pending: list[tuple[CodeChunkDocument, str]] = []

        def flush() -> None:
            if not pending:
                return
            texts = [text for _doc, text in pending]
            vectors = self._embedder(texts)
            if len(vectors) != len(pending):
                raise RuntimeError(
                    f"Embedder returned {len(vectors)} vectors for {len(pending)} chunks."
                )
            docs = []
            for (doc, _text), vector in zip(pending, vectors, strict=True):
                doc.embedding = vector
                docs.append(doc)
            if not CodeChunkDocument.bulk_insert(docs):
                raise RuntimeError(f"Failed to upsert {len(docs)} chunks into Qdrant.")
            stats.chunks_written += len(docs)
            pending.clear()

        for chunk in all_chunks:
            symbol = symbols_by_qn.get(chunk.metadata.qualified_name)
            prefix = context_by_qn.get(chunk.metadata.qualified_name)
            doc = CodeChunkDocument(
                id=deterministic_chunk_id(repo_id, chunk.metadata.qualified_name, chunk.part),
                content=chunk.content,
                repo_id=repo_id,
                commit_sha=commit_sha,
                qualified_name=chunk.metadata.qualified_name,
                file_path=chunk.metadata.file_path,
                start_line=chunk.metadata.start_line,
                end_line=chunk.metadata.end_line,
                language=chunk.metadata.language,
                parent_class=chunk.metadata.parent_class,
                part=chunk.part,
                total_parts=chunk.total_parts,
                symbol_kind=symbol.kind.value if symbol else None,
                signature=symbol.signature if symbol else None,
                docstring=symbol.docstring if symbol else None,
                contextual_prefix=prefix or None,
            )
            embedding_input = f"{prefix}\n{chunk.content}" if prefix else chunk.content
            pending.append((doc, embedding_input))
            if len(pending) >= self._batch_size:
                flush()

        flush()
        logger.info(f"Qdrant write complete: {stats.chunks_written} chunks.")
        return stats

    def _enrich_all(self, repo_id: str, chunks: list[CodeChunk]) -> dict[str, str]:
        from codeatlas.application.rag.contextual_enrichment import EnrichmentInput
        from codeatlas.application.rag.graph_context import build_graph_context_batch

        qualified_names = [c.metadata.qualified_name for c in chunks]
        graph_context = build_graph_context_batch(repo_id, qualified_names)

        items = [
            EnrichmentInput(
                qualified_name=c.metadata.qualified_name,
                content=c.content,
                graph_context=graph_context.get(c.metadata.qualified_name, ""),
            )
            for c in chunks
        ]
        return self._enricher.enrich_batch(items)
