from typing import Literal

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """Common result shape for dense/sparse/graph retrievers, so RRF can fuse them and
    the cross-encoder reranker can score them without caring which retriever found it.
    """

    content: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    source: Literal["dense", "sparse", "graph"]
    score: float
    docstring: str | None = None
    signature: str | None = None

    @property
    def dedup_key(self) -> str:
        """`qualified_name` is stable across retrievers; `content` is not (a Python
        string equality check would miss two retrievers finding the same symbol if
        whitespace differs), so fusion/dedup keys on this instead."""
        return self.qualified_name

    @property
    def source_ref(self) -> dict:
        """Every tool/context assembly step needs `[file.py:12-30]`-style citation —
        keep the shape in one place."""
        return {"file_path": self.file_path, "start": self.start_line, "end": self.end_line}


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Assemble retrieved chunks into a context block for the generation prompt, each
    one tagged with its citation so the LLM can (and must) quote `[file.py:12-30]`."""
    parts = []
    for i, chunk in enumerate(chunks):
        citation = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        header = f"[{i + 1}] {chunk.qualified_name}  ({citation})"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)
