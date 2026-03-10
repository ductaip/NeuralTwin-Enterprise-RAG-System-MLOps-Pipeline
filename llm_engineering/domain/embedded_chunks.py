from abc import ABC

from pydantic import UUID4, Field

from llm_engineering.domain.types import DataCategory

from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list[float] | None
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)

    # Hierarchical Retrieval: stores the parent (larger) document content.
    # Child chunks are used for precise vector search; parent content is used for LLM generation.
    parent_content: str | None = None

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        """
        Assembles context string from chunks.
        Uses Hierarchical Retrieval (Small-to-Big): if a chunk has parent_content,
        the full parent document is used instead of the small chunk, giving the LLM
        richer context while maintaining precise vector search via child chunks.
        """
        context = ""
        seen_parents = set()  # Deduplicate parent documents
        for i, chunk in enumerate(chunks):
            # Small-to-Big: prefer parent_content over chunk content
            if chunk.parent_content and chunk.document_id not in seen_parents:
                seen_parents.add(chunk.document_id)
                context += f"""
            Document {i + 1} (Expanded from Parent):
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Author: {chunk.author_full_name}
            Content: {chunk.parent_content}\n
            """
            elif chunk.document_id not in seen_parents:
                context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Author: {chunk.author_full_name}
            Content: {chunk.content}\n
            """

        return context


class EmbeddedPostChunk(EmbeddedChunk):
    class Config:
        name = "embedded_posts"
        category = DataCategory.POSTS
        use_vector_index = True


class EmbeddedArticleChunk(EmbeddedChunk):
    link: str

    class Config:
        name = "embedded_articles"
        category = DataCategory.ARTICLES
        use_vector_index = True


class EmbeddedRepositoryChunk(EmbeddedChunk):
    name: str
    link: str

    class Config:
        name = "embedded_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = True
