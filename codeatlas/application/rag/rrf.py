"""Reciprocal Rank Fusion. Kept as its own function so Phase 5 can unit-test and
ablate it independently of which retrievers feed it."""

from __future__ import annotations

from codeatlas.application.rag.models import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists by RRF score: sum(1 / (k + rank)) across lists.

    Dedupes on `qualified_name` (the same symbol found by two retrievers keeps the
    higher-scoring instance's content) rather than raw text, since dense/sparse/graph
    don't necessarily produce byte-identical strings for the same symbol.
    """
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = chunk.dedup_key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best_chunk or chunk.score > best_chunk[key].score:
                best_chunk[key] = chunk

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [best_chunk[key] for key in ordered_keys]
