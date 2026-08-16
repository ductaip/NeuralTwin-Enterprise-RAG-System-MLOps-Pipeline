"""`QdrantChunkWriter` with the Qdrant client patched out.

These test the mapping from `chunk_module` output to `CodeChunkDocument` — batching,
metadata carried through, embedding attached to the right chunk — not Qdrant itself.
Live behaviour against a real Qdrant instance is verified separately (see
`docs/PHASE2_RESULTS.md`), not in the unit suite, matching how `test_symbol_resolver.py`
covers logic and the ingest CLI covers the real Neo4j write.
"""

from __future__ import annotations

from unittest.mock import patch

from codeatlas.domain.code.chunk import CodeChunkDocument
from codeatlas.ingestion.models import SourceFile
from codeatlas.ingestion.python_parser import PythonParser
from codeatlas.ingestion.qdrant_writer import QdrantChunkWriter, deterministic_chunk_id
from codeatlas.ingestion.symbol_resolver import SymbolResolver


def test_deterministic_chunk_id_is_stable_and_distinguishes_inputs():
    a = deterministic_chunk_id("repo1", "pkg.mod.func", 0)
    assert a == deterministic_chunk_id("repo1", "pkg.mod.func", 0)
    assert a != deterministic_chunk_id("repo1", "pkg.mod.func", 1)
    assert a != deterministic_chunk_id("repo1", "pkg.mod.other", 0)
    assert a != deterministic_chunk_id("repo2", "pkg.mod.func", 0)


def test_reingesting_the_same_repo_reuses_ids_instead_of_duplicating():
    """This is the property that makes Qdrant writes idempotent, matching graph_builder's
    MERGE-on-key behaviour: re-running ingest overwrites points instead of piling up
    duplicates with new random UUIDs."""
    pairs, symbols = build({"pkg/mod.py": "def a():\n    return 1\n"})
    writer = QdrantChunkWriter(embedder=fake_embedder, batch_size=64)

    captured_first: list[CodeChunkDocument] = []
    with patch.object(CodeChunkDocument, "get_or_create_collection"), patch.object(
        CodeChunkDocument, "bulk_insert", side_effect=lambda docs: captured_first.extend(docs) or True
    ):
        writer.write(repo_id="repo1", commit_sha="v1", language="python", modules=pairs, symbols_by_qn=symbols)

    captured_second: list[CodeChunkDocument] = []
    with patch.object(CodeChunkDocument, "get_or_create_collection"), patch.object(
        CodeChunkDocument, "bulk_insert", side_effect=lambda docs: captured_second.extend(docs) or True
    ):
        writer.write(repo_id="repo1", commit_sha="v2", language="python", modules=pairs, symbols_by_qn=symbols)

    assert captured_first[0].id == captured_second[0].id


def build(files: dict[str, str]):
    parser = PythonParser()
    pairs = []
    for path, content in files.items():
        module_name = path[:-3].replace("/", ".")
        source = SourceFile(
            path=path, module_name=module_name, content=content, sha="s", loc=content.count("\n") + 1
        )
        pairs.append((source, parser.parse(source)))
    modules = [p for _s, p in pairs]
    resolver = SymbolResolver(modules)
    resolver.resolve_all()
    return pairs, {s.qualified_name: s for s in resolver.symbol_table.values()}


def fake_embedder(texts: list[str]) -> list[list[float]]:
    return [[float(len(t))] for t in texts]


def test_batches_and_embeds_every_chunk():
    pairs, symbols = build(
        {"pkg/mod.py": "def a():\n    return 1\n\n\ndef b():\n    return 2\n"}
    )
    writer = QdrantChunkWriter(embedder=fake_embedder, batch_size=1)

    with patch.object(CodeChunkDocument, "get_or_create_collection"), patch.object(
        CodeChunkDocument, "bulk_insert", return_value=True
    ) as bulk_insert:
        stats = writer.write(
            repo_id="repo1", commit_sha="abc", language="python", modules=pairs, symbols_by_qn=symbols
        )

    assert stats.chunks_written == 2
    assert bulk_insert.call_count == 2  # batch_size=1 forces a flush per chunk


def test_metadata_and_embedding_carried_through():
    pairs, symbols = build({"pkg/mod.py": "def alpha():\n    return 1\n"})
    writer = QdrantChunkWriter(embedder=fake_embedder, batch_size=64)

    captured: list[CodeChunkDocument] = []

    def capture(docs):
        captured.extend(docs)
        return True

    with patch.object(CodeChunkDocument, "get_or_create_collection"), patch.object(
        CodeChunkDocument, "bulk_insert", side_effect=capture
    ):
        writer.write(
            repo_id="repo1", commit_sha="abc123", language="python", modules=pairs, symbols_by_qn=symbols
        )

    assert len(captured) == 1
    doc = captured[0]
    assert doc.qualified_name == "pkg.mod.alpha"
    assert doc.repo_id == "repo1"
    assert doc.commit_sha == "abc123"
    assert doc.language == "python"
    assert doc.symbol_kind == "function"
    assert doc.embedding == [float(len(doc.content))]


def test_symbol_lookup_miss_leaves_optional_fields_none():
    """A chunk whose qualified_name is not in the symbol table still gets written."""
    pairs, _symbols = build({"pkg/mod.py": "def alpha():\n    return 1\n"})
    writer = QdrantChunkWriter(embedder=fake_embedder, batch_size=64)

    captured: list[CodeChunkDocument] = []
    with patch.object(CodeChunkDocument, "get_or_create_collection"), patch.object(
        CodeChunkDocument, "bulk_insert", side_effect=lambda docs: captured.extend(docs) or True
    ):
        writer.write(
            repo_id="repo1", commit_sha=None, language="python", modules=pairs, symbols_by_qn={}
        )

    assert captured[0].symbol_kind is None
    assert captured[0].signature is None


def test_no_chunks_does_not_call_bulk_insert():
    pairs, symbols = build({"pkg/mod.py": "class Empty:\n    pass\n"})
    writer = QdrantChunkWriter(embedder=fake_embedder, batch_size=64)

    with patch.object(CodeChunkDocument, "get_or_create_collection"), patch.object(
        CodeChunkDocument, "bulk_insert"
    ) as bulk_insert:
        stats = writer.write(
            repo_id="repo1", commit_sha=None, language="python", modules=pairs, symbols_by_qn=symbols
        )

    assert stats.chunks_written == 0
    bulk_insert.assert_not_called()


def test_bulk_insert_failure_raises():
    pairs, symbols = build({"pkg/mod.py": "def a():\n    return 1\n"})
    writer = QdrantChunkWriter(embedder=fake_embedder, batch_size=64)

    with patch.object(CodeChunkDocument, "get_or_create_collection"), patch.object(
        CodeChunkDocument, "bulk_insert", return_value=False
    ):
        try:
            writer.write(
                repo_id="repo1", commit_sha=None, language="python", modules=pairs, symbols_by_qn=symbols
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError on failed upsert")
