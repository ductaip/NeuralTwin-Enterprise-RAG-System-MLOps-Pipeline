"""Write the code graph to Neo4j.

Two properties matter and are tested by re-running the same commit:

* **Batched** — every write goes through one `UNWIND` per batch, never a query per node.
* **Idempotent** — every write is a `MERGE` on `(repo_id, qualified_name)`, so a second run
  over the same commit updates properties in place and creates nothing new.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from codeatlas.ingestion.models import (
    CallEdge,
    InheritanceEdge,
    SourceFile,
    SymbolDef,
    SymbolKind,
    TestEdge,
)
from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter
from codeatlas.settings import settings

INDEX_STATEMENTS = [
    "CREATE INDEX fn_qn IF NOT EXISTS FOR (f:Function) ON (f.qualified_name)",
    "CREATE INDEX cls_qn IF NOT EXISTS FOR (c:Class) ON (c.qualified_name)",
    "CREATE INDEX file_p IF NOT EXISTS FOR (f:File) ON (f.path)",
    "CREATE INDEX test_qn IF NOT EXISTS FOR (t:Test) ON (t.qualified_name)",
    "CREATE INDEX mod_qn IF NOT EXISTS FOR (m:Module) ON (m.qualified_name)",
    # The MERGE key is the pair, so it gets its own composite index.
    "CREATE INDEX fn_repo_qn IF NOT EXISTS FOR (f:Function) ON (f.repo_id, f.qualified_name)",
    "CREATE INDEX cls_repo_qn IF NOT EXISTS FOR (c:Class) ON (c.repo_id, c.qualified_name)",
]

MERGE_REPOSITORY = """
MERGE (r:Repository {repo_id: $repo_id})
SET r.name = $name, r.url = $url, r.commit_sha = $commit_sha,
    r.indexed_at = datetime(), r.language = $language
"""

MERGE_FILES = """
UNWIND $rows AS row
MERGE (f:File {repo_id: $repo_id, path: row.path})
SET f.language = row.language, f.loc = row.loc, f.sha = row.sha
WITH f
MATCH (r:Repository {repo_id: $repo_id})
MERGE (f)-[:IN_REPO]->(r)
"""

MERGE_MODULES = """
UNWIND $rows AS row
MERGE (m:Module {repo_id: $repo_id, qualified_name: row.qualified_name})
SET m.name = row.name
"""

MERGE_CLASSES = """
UNWIND $rows AS row
MERGE (c:Class {repo_id: $repo_id, qualified_name: row.qualified_name})
SET c.name = row.name, c.docstring = row.docstring,
    c.start_line = row.start_line, c.end_line = row.end_line,
    c.is_public = row.is_public, c.file_path = row.file_path
WITH c, row
MATCH (f:File {repo_id: $repo_id, path: row.file_path})
MERGE (c)-[:DEFINED_IN]->(f)
"""

MERGE_FUNCTIONS = """
UNWIND $rows AS row
MERGE (fn:Function {repo_id: $repo_id, qualified_name: row.qualified_name})
SET fn.name = row.name, fn.signature = row.signature, fn.docstring = row.docstring,
    fn.start_line = row.start_line, fn.end_line = row.end_line,
    fn.is_async = row.is_async, fn.is_public = row.is_public,
    fn.complexity = row.complexity, fn.file_path = row.file_path,
    fn.is_test = row.is_test
WITH fn, row
MATCH (f:File {repo_id: $repo_id, path: row.file_path})
MERGE (fn)-[:DEFINED_IN]->(f)
"""

MARK_TESTS = """
UNWIND $rows AS row
MATCH (fn:Function {repo_id: $repo_id, qualified_name: row.qualified_name})
SET fn:Test, fn.framework = row.framework
"""

MERGE_METHOD_OF = """
UNWIND $rows AS row
MATCH (fn:Function {repo_id: $repo_id, qualified_name: row.qualified_name})
MATCH (c:Class {repo_id: $repo_id, qualified_name: row.parent_class})
MERGE (fn)-[:METHOD_OF]->(c)
"""

MERGE_IMPORTS = """
UNWIND $rows AS row
MATCH (f:File {repo_id: $repo_id, path: row.file_path})
MERGE (m:Module {repo_id: $repo_id, qualified_name: row.module})
MERGE (f)-[i:IMPORTS]->(m)
SET i.alias = row.alias
"""

MERGE_CALLS = """
UNWIND $rows AS row
MATCH (caller {repo_id: $repo_id, qualified_name: row.caller_qn})
MATCH (callee {repo_id: $repo_id, qualified_name: row.callee_qn})
MERGE (caller)-[c:CALLS {line: row.line}]->(callee)
SET c.confidence = row.confidence, c.reason = row.reason
"""

MERGE_INHERITS = """
UNWIND $rows AS row
MATCH (child:Class {repo_id: $repo_id, qualified_name: row.child_qn})
MATCH (parent:Class {repo_id: $repo_id, qualified_name: row.parent_qn})
MERGE (child)-[:INHERITS]->(parent)
"""

MERGE_TESTS_EDGE = """
UNWIND $rows AS row
MATCH (t:Function {repo_id: $repo_id, qualified_name: row.test_qn})
MATCH (target:Function {repo_id: $repo_id, qualified_name: row.target_qn})
MERGE (t)-[r:TESTS]->(target)
SET r.confidence = row.confidence
"""


@dataclass
class BuildStats:
    nodes_created: int = 0
    relationships_created: int = 0
    properties_set: int = 0

    def add(self, counters: dict[str, int]) -> None:
        self.nodes_created += counters.get("nodes_created", 0)
        self.relationships_created += counters.get("relationships_created", 0)
        self.properties_set += counters.get("properties_set", 0)


class GraphBuilder:
    def __init__(self, adapter: Neo4jAdapter | None = None, batch_size: int | None = None):
        self.adapter = adapter or Neo4jAdapter()
        self.batch_size = batch_size or settings.INGESTION_BATCH_SIZE

    def ensure_indexes(self) -> None:
        for statement in INDEX_STATEMENTS:
            self.adapter.execute_write(statement)
        logger.info(f"Ensured {len(INDEX_STATEMENTS)} Neo4j indexes.")

    def build(
        self,
        repo_id: str,
        files: list[SourceFile],
        symbols: list[SymbolDef],
        call_edges: list[CallEdge],
        inheritance_edges: list[InheritanceEdge],
        test_edges: list[TestEdge],
        imports: list[dict],
        url: str = "",
        commit_sha: str | None = None,
        language: str = "python",
    ) -> BuildStats:
        stats = BuildStats()
        self.ensure_indexes()

        stats.add(
            self.adapter.execute_write(
                MERGE_REPOSITORY,
                {
                    "repo_id": repo_id,
                    "name": repo_id,
                    "url": url,
                    "commit_sha": commit_sha,
                    "language": language,
                },
            )
        )

        stats.add(
            self._batch(
                MERGE_FILES,
                [
                    {"path": f.path, "language": language, "loc": f.loc, "sha": f.sha}
                    for f in files
                ],
                repo_id,
            )
        )

        modules = [s for s in symbols if s.kind is SymbolKind.MODULE]
        classes = [s for s in symbols if s.kind is SymbolKind.CLASS]
        functions = [s for s in symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)]

        stats.add(
            self._batch(
                MERGE_MODULES,
                [{"qualified_name": s.qualified_name, "name": s.name} for s in modules],
                repo_id,
            )
        )
        stats.add(
            self._batch(
                MERGE_CLASSES,
                [
                    {
                        "qualified_name": s.qualified_name,
                        "name": s.name,
                        "docstring": s.docstring,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "is_public": s.is_public,
                        "file_path": s.file_path,
                    }
                    for s in classes
                ],
                repo_id,
            )
        )
        stats.add(
            self._batch(
                MERGE_FUNCTIONS,
                [
                    {
                        "qualified_name": s.qualified_name,
                        "name": s.name,
                        "signature": s.signature,
                        "docstring": s.docstring,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "is_async": s.is_async,
                        "is_public": s.is_public,
                        "is_test": s.is_test,
                        "complexity": s.complexity,
                        "file_path": s.file_path,
                    }
                    for s in functions
                ],
                repo_id,
            )
        )

        stats.add(
            self._batch(
                MARK_TESTS,
                [
                    {"qualified_name": s.qualified_name, "framework": "pytest"}
                    for s in functions
                    if s.is_test
                ],
                repo_id,
            )
        )
        stats.add(
            self._batch(
                MERGE_METHOD_OF,
                [
                    {"qualified_name": s.qualified_name, "parent_class": s.parent_class}
                    for s in functions
                    if s.parent_class
                ],
                repo_id,
            )
        )
        stats.add(self._batch(MERGE_IMPORTS, imports, repo_id))
        stats.add(
            self._batch(
                MERGE_CALLS,
                [
                    {
                        "caller_qn": e.caller_qn,
                        "callee_qn": e.callee_qn,
                        "line": e.line,
                        "confidence": e.confidence,
                        "reason": e.reason,
                    }
                    for e in call_edges
                ],
                repo_id,
            )
        )
        stats.add(
            self._batch(
                MERGE_INHERITS,
                [
                    {"child_qn": e.child_qn, "parent_qn": e.parent_qn}
                    for e in inheritance_edges
                    if e.status.value == "internal"
                ],
                repo_id,
            )
        )
        stats.add(
            self._batch(
                MERGE_TESTS_EDGE,
                [
                    {"test_qn": e.test_qn, "target_qn": e.target_qn, "confidence": e.confidence}
                    for e in test_edges
                ],
                repo_id,
            )
        )

        logger.info(
            f"Graph build complete: +{stats.nodes_created} nodes, "
            f"+{stats.relationships_created} relationships."
        )
        return stats

    def _batch(self, query: str, rows: list[dict], repo_id: str) -> dict[str, int]:
        if not rows:
            return {}
        return self.adapter.execute_write_batch(
            query, rows, batch_size=self.batch_size, extra_params={"repo_id": repo_id}
        )
