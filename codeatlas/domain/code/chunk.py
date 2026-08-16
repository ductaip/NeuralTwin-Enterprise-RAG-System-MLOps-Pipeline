import uuid
from uuid import UUID

from pydantic import Field

from codeatlas.domain.base import VectorBaseDocument
from codeatlas.domain.types import DataCategory


class CodeChunkDocument(VectorBaseDocument):
    """One function/method chunk, embedded and stored in Qdrant.

    Mirrors `codeatlas.ingestion.models.CodeChunk` but flattened (Qdrant payload fields,
    not a nested `metadata` object) so retrievers can filter on `qualified_name`/`repo_id`
    directly. `qualified_name` matches the Neo4j `Function.qualified_name` key, which is
    what lets `GraphRetriever` hydrate content for symbols it finds by graph traversal.
    """

    # Overrides `VectorBaseDocument.id: UUID4` (pydantic strictly checks version == 4).
    # `qdrant_writer.py` assigns a version-5 UUID derived from (repo_id, qualified_name,
    # part) so re-ingesting a repo overwrites points instead of duplicating them.
    id: UUID = Field(default_factory=uuid.uuid4)

    content: str
    embedding: list[float] | None = None

    repo_id: str
    commit_sha: str | None = None
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    parent_class: str | None = None
    part: int = 0
    total_parts: int = 1
    symbol_kind: str | None = None
    signature: str | None = None
    docstring: str | None = None
    contextual_prefix: str | None = None
    """LLM-generated one-sentence context (module/role/callers), prepended to `content`
    before embedding. Kept as a separate field rather than mutated into `content` so
    citations and generation context still show the real code, not a sentence glued
    onto it. See `codeatlas.application.rag.contextual_enrichment`."""

    class Config:
        name = "code_chunks"
        category = DataCategory.CODE_CHUNKS
