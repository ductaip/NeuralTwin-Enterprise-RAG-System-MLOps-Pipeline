"""Build the "module / role / callers" facts contextual enrichment needs, from the graph
built in Phase 1 — batched via UNWIND rather than one Cypher round-trip per symbol.
"""

from __future__ import annotations

from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter

_BATCH_CONTEXT_CYPHER = """
UNWIND $qualified_names AS qn
MATCH (f:Function {repo_id: $repo_id, qualified_name: qn})
OPTIONAL MATCH (f)-[:DEFINED_IN]->(file:File)
OPTIONAL MATCH (f)-[:METHOD_OF]->(cls:Class)
OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
WITH qn, file.path AS file_path, cls.qualified_name AS parent_class,
     collect(DISTINCT caller.qualified_name)[0..5] AS callers
RETURN qn, file_path, parent_class, callers
"""


def build_graph_context_batch(
    repo_id: str, qualified_names: list[str], adapter: Neo4jAdapter | None = None
) -> dict[str, str]:
    """Returns {qualified_name: one-line fact string} for use in the enrichment prompt."""
    if not qualified_names:
        return {}
    adapter = adapter or Neo4jAdapter()
    rows = adapter.execute_read(
        _BATCH_CONTEXT_CYPHER, {"repo_id": repo_id, "qualified_names": qualified_names}
    )

    context: dict[str, str] = {}
    for row in rows:
        parts = []
        if row.get("file_path"):
            parts.append(f"defined in {row['file_path']}")
        if row.get("parent_class"):
            parts.append(f"method of {row['parent_class']}")
        callers = row.get("callers") or []
        if callers:
            parts.append(f"called by {', '.join(callers)}")
        else:
            parts.append("no known callers in this repo")
        context[row["qn"]] = "; ".join(parts) if parts else "no graph facts available"

    for qn in qualified_names:
        context.setdefault(qn, "no graph facts available")
    return context
