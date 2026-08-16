"""Graph-driven retrieval: find symbols by name/docstring match in Neo4j, then hydrate
their source text from Qdrant.

Implements the same interface as `DenseCodeRetriever`/`SparseCodeRetriever` so RRF can
fuse all three. This is the retriever the vector-only ablation baseline is expected to
lose to on structural queries (spec §3.3 Bảng A hypothesis) — worth keeping literal
identifier matching here rather than anything embedding-based, or the ablation stops
measuring what it claims to.
"""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchAny

from codeatlas.application.rag.identifier_tokenizer import tokenize
from codeatlas.application.rag.models import RetrievedChunk
from codeatlas.domain.code.chunk import CodeChunkDocument
from codeatlas.infrastructure.db.qdrant import connection
from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter

_SEARCH_CYPHER = """
MATCH (f:Function {repo_id: $repo_id})
WITH f, size([t IN $tokens WHERE
    toLower(f.name) CONTAINS t OR toLower(coalesce(f.docstring, '')) CONTAINS t
]) AS match_count
WHERE match_count > 0
OPTIONAL MATCH (f)<-[:CALLS]-(caller)
WITH f, match_count, count(DISTINCT caller) AS fan_in
RETURN f.qualified_name AS qualified_name, f.file_path AS file_path,
       f.start_line AS start_line, f.end_line AS end_line,
       f.docstring AS docstring, f.signature AS signature,
       match_count, fan_in
ORDER BY match_count DESC, fan_in DESC
LIMIT $limit
"""

_MAX_QUERY_TOKENS = 8


class GraphRetriever:
    def __init__(self, repo_id: str, adapter: Neo4jAdapter | None = None):
        self.repo_id = repo_id
        self.adapter = adapter or Neo4jAdapter()

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        tokens = sorted(set(tokenize(query)), key=len, reverse=True)[:_MAX_QUERY_TOKENS]
        if not tokens:
            return []

        rows = self.adapter.execute_read(
            _SEARCH_CYPHER,
            {"repo_id": self.repo_id, "tokens": tokens, "limit": k},
        )
        if not rows:
            return []

        content_by_qn = self._hydrate_content([r["qualified_name"] for r in rows])
        max_score = max((r["match_count"] for r in rows), default=1) or 1

        results: list[RetrievedChunk] = []
        for row in rows:
            qn = row["qualified_name"]
            content = content_by_qn.get(qn)
            if content is None:
                # No embedded chunk for this symbol (e.g. it produced no chunk at
                # ingest time) — fall back to signature/docstring rather than dropping
                # a graph hit the caller asked for.
                content = row["signature"] or qn
                if row["docstring"]:
                    content = f"{content}\n{row['docstring']}"

            results.append(
                RetrievedChunk(
                    content=content,
                    qualified_name=qn,
                    file_path=row["file_path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    source="graph",
                    score=row["match_count"] / max_score,
                    docstring=row["docstring"],
                    signature=row["signature"],
                )
            )
        return results

    def _hydrate_content(self, qualified_names: list[str]) -> dict[str, str]:
        if not qualified_names:
            return {}
        records, _offset = connection.scroll(
            collection_name=CodeChunkDocument.get_collection_name(),
            scroll_filter=Filter(
                must=[FieldCondition(key="qualified_name", match=MatchAny(any=qualified_names))]
            ),
            limit=len(qualified_names),
            with_payload=True,
            with_vectors=False,
        )
        return {r.payload["qualified_name"]: r.payload["content"] for r in records}
