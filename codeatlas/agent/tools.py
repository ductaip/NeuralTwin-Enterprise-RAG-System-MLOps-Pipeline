"""Seven QA-mode tools, spec §2.4. Shared verbatim by both orchestrators
(`custom_react.py`, `langgraph_qa.py`) — if they used different tools or different
retrieval, Bảng B would measure the tool set, not the orchestrator.

Harness rules (enforced by the *caller* — the tool-call ceiling and loop detection live
in the orchestrator loop, not here, since both orchestrators share this module):
- every tool returns a JSON-serialisable dict with a `source` (single-result tools) or
  each item in `results`/`callers`/etc. carrying its own `source`
- errors are actionable: `search_symbol` and any qualified-name lookup that misses
  suggests the closest known name via difflib, never a bare "not found"
- nothing here invents a citation — a miss returns `{"error": ...}`, not a guess
"""

from __future__ import annotations

import difflib
from typing import Any, Callable

from codeatlas.application.rag.retriever import ContextRetriever
from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter
from codeatlas.settings import settings

TOOL_NAMES = (
    "search_semantic",
    "search_symbol",
    "get_callers",
    "get_callees",
    "impact_analysis",
    "read_source",
    "list_module_structure",
)


def _source(file_path: str, start: int, end: int) -> dict:
    return {"file_path": file_path, "start": start, "end": end}


def _closest_symbol_suggestion(adapter: Neo4jAdapter, repo_id: str, name: str) -> dict:
    """difflib-ranked closest known qualified_name — the actionable half of an error."""
    rows = adapter.execute_read(
        "MATCH (f:Function {repo_id: $repo_id}) RETURN f.qualified_name AS qn",
        {"repo_id": repo_id},
    )
    candidates = [r["qn"] for r in rows]
    if not candidates:
        return {"error": f"Symbol {name!r} not found. Repo {repo_id!r} has no indexed functions."}

    matches = difflib.get_close_matches(name, candidates, n=3, cutoff=0.4)
    if not matches:
        # cutoff too strict for a very short/unrelated query — fall back to best-effort
        scored = sorted(
            candidates, key=lambda c: difflib.SequenceMatcher(None, name, c).ratio(), reverse=True
        )
        matches = scored[:3]

    best = matches[0]
    score = difflib.SequenceMatcher(None, name, best).ratio()
    return {
        "error": f"Symbol {name!r} not found. Closest: {best!r} ({score:.2f})",
        "suggestions": matches,
    }


class AgentTools:
    """Bound to one `repo_id`. Constructed once per query by whichever orchestrator
    is running; both share this exact class."""

    def __init__(self, repo_id: str, adapter: Neo4jAdapter | None = None):
        self.repo_id = repo_id
        self.adapter = adapter or Neo4jAdapter()
        self._retriever: ContextRetriever | None = None

    @property
    def retriever(self) -> ContextRetriever:
        if self._retriever is None:
            self._retriever = ContextRetriever(repo_id=self.repo_id)
        return self._retriever

    def as_dispatch_table(self) -> dict[str, Callable[..., dict]]:
        return {name: getattr(self, name) for name in TOOL_NAMES}

    # -- 1. search_semantic -------------------------------------------------------

    def search_semantic(self, query: str, top_k: int = 5) -> dict:
        chunks = self.retriever.search(query, k=top_k)
        return {
            "results": [
                {
                    "qualified_name": c.qualified_name,
                    "content": c.content,
                    "source": _source(c.file_path, c.start_line, c.end_line),
                }
                for c in chunks
            ]
        }

    # -- 2. search_symbol -----------------------------------------------------------

    def search_symbol(self, name: str) -> dict:
        rows = self.adapter.execute_read(
            """
            MATCH (f {repo_id: $repo_id})
            WHERE (f:Function OR f:Class) AND (f.qualified_name = $name OR f.name = $name)
            RETURN f.qualified_name AS qualified_name, labels(f) AS labels,
                   f.signature AS signature, f.docstring AS docstring,
                   f.file_path AS file_path, f.start_line AS start_line, f.end_line AS end_line
            LIMIT 5
            """,
            {"repo_id": self.repo_id, "name": name},
        )
        if not rows:
            return _closest_symbol_suggestion(self.adapter, self.repo_id, name)

        return {
            "results": [
                {
                    "qualified_name": r["qualified_name"],
                    "kind": "class" if "Class" in r["labels"] else "function",
                    "signature": r["signature"],
                    "docstring": r["docstring"],
                    "source": _source(r["file_path"], r["start_line"], r["end_line"]),
                }
                for r in rows
            ]
        }

    # -- 3/4. get_callers / get_callees ----------------------------------------------

    def get_callers(self, qualified_name: str, depth: int = 1) -> dict:
        return self._call_neighbors(qualified_name, depth, direction="callers")

    def get_callees(self, qualified_name: str, depth: int = 1) -> dict:
        return self._call_neighbors(qualified_name, depth, direction="callees")

    def _call_neighbors(self, qualified_name: str, depth: int, direction: str) -> dict:
        depth = max(1, min(depth, 5))
        if not self._symbol_exists(qualified_name):
            return _closest_symbol_suggestion(self.adapter, self.repo_id, qualified_name)

        pattern = (
            f"(neighbor:Function)-[c:CALLS*1..{depth}]->(f:Function {{repo_id: $repo_id, qualified_name: $qn}})"
            if direction == "callers"
            else f"(f:Function {{repo_id: $repo_id, qualified_name: $qn}})-[c:CALLS*1..{depth}]->(neighbor:Function)"
        )
        rows = self.adapter.execute_read(
            f"""
            MATCH {pattern}
            WHERE all(r IN c WHERE r.confidence >= $min_confidence)
            WITH DISTINCT neighbor, min(length(c)) AS distance
            RETURN neighbor.qualified_name AS qualified_name, neighbor.signature AS signature,
                   neighbor.file_path AS file_path, neighbor.start_line AS start_line,
                   neighbor.end_line AS end_line, distance
            ORDER BY distance
            LIMIT 50
            """,
            {
                "repo_id": self.repo_id,
                "qn": qualified_name,
                "min_confidence": settings.CALL_EDGE_MIN_CONFIDENCE_IMPACT,
            },
        )
        key = "callers" if direction == "callers" else "callees"
        return {
            key: [
                {
                    "qualified_name": r["qualified_name"],
                    "signature": r["signature"],
                    "distance": r["distance"],
                    "source": _source(r["file_path"], r["start_line"], r["end_line"]),
                }
                for r in rows
            ]
        }

    # -- 5. impact_analysis -----------------------------------------------------------

    def impact_analysis(self, qualified_name: str) -> dict:
        if not self._symbol_exists(qualified_name):
            return _closest_symbol_suggestion(self.adapter, self.repo_id, qualified_name)

        params = {
            "repo_id": self.repo_id,
            "qn": qualified_name,
            "min_confidence": settings.CALL_EDGE_MIN_CONFIDENCE_IMPACT,
        }

        impacted_rows = self.adapter.execute_read(
            """
            MATCH path = (impacted:Function {repo_id: $repo_id})-[c:CALLS*1..3]->
                          (f:Function {repo_id: $repo_id, qualified_name: $qn})
            WHERE all(r IN c WHERE r.confidence >= $min_confidence)
            WITH impacted, min(length(path)) AS distance
            RETURN impacted.qualified_name AS qualified_name, impacted.file_path AS file_path,
                   impacted.start_line AS start_line, impacted.end_line AS end_line, distance
            ORDER BY distance
            LIMIT 100
            """,
            params,
        )
        # Cypher [3]: affected tests, TESTS only in Phase 3 — COVERS lands in Phase 4.
        test_rows = self.adapter.execute_read(
            """
            MATCH (t:Test {repo_id: $repo_id})-[:TESTS]->(impacted:Function)-[c:CALLS*0..3]->
                  (f:Function {repo_id: $repo_id, qualified_name: $qn})
            WHERE all(r IN c WHERE r.confidence >= $min_confidence)
            RETURN DISTINCT t.qualified_name AS qualified_name, t.file_path AS file_path
            LIMIT 200
            """,
            params,
        )

        return {
            "impacted_symbols": [
                {
                    "qualified_name": r["qualified_name"],
                    "distance": r["distance"],
                    "source": _source(r["file_path"], r["start_line"], r["end_line"]),
                }
                for r in impacted_rows
            ],
            "affected_tests": [r["qualified_name"] for r in test_rows],
            "affected_tests_source": [
                {"qualified_name": r["qualified_name"], "file_path": r["file_path"]}
                for r in test_rows
            ],
        }

    # -- 6. read_source -----------------------------------------------------------

    def read_source(self, file_path: str, start: int, end: int) -> dict:
        """Serves from indexed `CodeChunkDocument` content, not a filesystem read.

        Phase 1's clone of a remote repo is deleted after ingest (`ingest.py`'s
        `cleanup_dir`), and there is no stored `repo_id -> checkout path` mapping for a
        local one either. Qdrant already holds the exact text of every indexed chunk,
        so this looks up chunks in `file_path` whose line range overlaps [start, end]
        and returns their concatenated content — the only source of source text this
        agent actually has, and one that can't silently go stale relative to what was
        indexed. A range with no covering chunk is a real miss, reported as one.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from codeatlas.domain.code.chunk import CodeChunkDocument
        from codeatlas.infrastructure.db.qdrant import connection

        records, _ = connection.scroll(
            collection_name=CodeChunkDocument.get_collection_name(),
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="repo_id", match=MatchValue(value=self.repo_id)),
                    FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                ]
            ),
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        overlapping = [
            r.payload
            for r in records
            if r.payload["start_line"] <= end and r.payload["end_line"] >= start
        ]
        if not overlapping:
            return {
                "error": (
                    f"No indexed chunk covers {file_path}:{start}-{end}. "
                    "Only ranges that were indexed as a function/method chunk can be read."
                )
            }

        overlapping.sort(key=lambda p: (p["start_line"], p.get("part", 0)))
        content = "\n".join(p["content"] for p in overlapping)
        real_start = min(p["start_line"] for p in overlapping)
        real_end = max(p["end_line"] for p in overlapping)
        return {"content": content, "source": _source(file_path, real_start, real_end)}

    # -- 7. list_module_structure -----------------------------------------------------

    def list_module_structure(self, module: str) -> dict:
        prefix = f"{module}."
        rows = self.adapter.execute_read(
            """
            MATCH (n {repo_id: $repo_id})
            WHERE (n:Function OR n:Class) AND n.qualified_name STARTS WITH $prefix
            RETURN n.qualified_name AS qualified_name, labels(n) AS labels,
                   n.file_path AS file_path, n.start_line AS start_line, n.end_line AS end_line,
                   n.docstring AS docstring
            """,
            {"repo_id": self.repo_id, "prefix": prefix},
        )
        if not rows:
            return {
                "error": f"No symbol found under module {module!r}. "
                "Check the module path matches an indexed file's dotted name."
            }

        classes, functions = [], []
        for r in rows:
            remainder = r["qualified_name"][len(prefix) :]
            if "." in remainder:
                continue  # nested (method of a class already listed, or deeper module)
            entry = {
                "qualified_name": r["qualified_name"],
                "docstring": r["docstring"],
                "source": _source(r["file_path"], r["start_line"], r["end_line"]),
            }
            (classes if "Class" in r["labels"] else functions).append(entry)

        return {"module": module, "classes": classes, "functions": functions}

    # -- helpers -----------------------------------------------------------------------

    def _symbol_exists(self, qualified_name: str) -> bool:
        rows = self.adapter.execute_read(
            "MATCH (f:Function {repo_id: $repo_id, qualified_name: $qn}) RETURN f LIMIT 1",
            {"repo_id": self.repo_id, "qn": qualified_name},
        )
        return bool(rows)
