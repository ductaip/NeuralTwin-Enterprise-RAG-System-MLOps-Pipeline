from __future__ import annotations

from codeatlas.ingestion.chunker import chunk_module, chunk_symbol
from codeatlas.ingestion.models import SourceFile
from codeatlas.ingestion.python_parser import PythonParser


def parse(content: str, path: str = "pkg/mod.py") -> tuple[SourceFile, object]:
    source = SourceFile(
        path=path,
        module_name=path[:-3].replace("/", "."),
        content=content,
        sha="sha",
        loc=content.count("\n") + 1,
    )
    return source, PythonParser().parse(source)


def test_one_chunk_per_function_includes_docstring():
    source, parsed = parse(
        'def alpha():\n    """Doc for alpha."""\n    return 1\n\n\ndef beta():\n    return 2\n'
    )
    chunks = chunk_module(parsed, source, repo_id="r")
    assert len(chunks) == 2
    alpha = next(c for c in chunks if c.metadata.qualified_name == "pkg.mod.alpha")
    assert "Doc for alpha." in alpha.content
    assert "def beta" not in alpha.content


def test_metadata_carries_parent_class_and_lines():
    source, parsed = parse("class Engine:\n    def start(self):\n        return 1\n")
    chunks = chunk_module(parsed, source, repo_id="repo-1")
    start = next(c for c in chunks if c.metadata.qualified_name == "pkg.mod.Engine.start")
    assert start.metadata.parent_class == "pkg.mod.Engine"
    assert start.metadata.repo_id == "repo-1"
    assert start.metadata.language == "python"
    assert start.metadata.start_line == 2


def test_classes_and_modules_do_not_become_chunks():
    source, parsed = parse("class Engine:\n    pass\n")
    assert chunk_module(parsed, source, repo_id="r") == []


def test_long_function_splits_and_every_part_keeps_the_header():
    body = "\n".join(f"    x{i} = {i}" for i in range(120))
    source, parsed = parse(f"def big():\n{body}\n")
    symbol = next(s for s in parsed.symbols if s.qualified_name == "pkg.mod.big")

    chunks = chunk_symbol(symbol, source, repo_id="r", max_lines=40)
    assert len(chunks) > 1
    assert all(c.total_parts == len(chunks) for c in chunks)
    # A chunk retrieved on its own still says which function it belongs to.
    assert all("def big():" in c.content for c in chunks)


def test_short_function_is_a_single_chunk():
    source, parsed = parse("def small():\n    return 1\n")
    symbol = next(s for s in parsed.symbols if s.qualified_name == "pkg.mod.small")
    chunks = chunk_symbol(symbol, source, repo_id="r", max_lines=100)
    assert len(chunks) == 1
    assert chunks[0].part == 0 and chunks[0].total_parts == 1
