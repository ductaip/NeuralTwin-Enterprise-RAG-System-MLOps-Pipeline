"""Turn parsed symbols into retrieval chunks.

One chunk is one function or method, docstring included. Functions longer than
`max_lines` are split on top-level statement boundaries, and every part repeats the
signature line so a chunk retrieved in isolation still says what it belongs to.
"""

from __future__ import annotations

import ast

from codeatlas.ingestion.models import (
    ChunkMetadata,
    CodeChunk,
    ParsedModule,
    SourceFile,
    SymbolDef,
    SymbolKind,
)

DEFAULT_MAX_LINES = 100


def _statement_boundaries(source_lines: list[str], symbol: SymbolDef) -> list[int]:
    """Line offsets (0-based, relative to the symbol) where a top-level statement starts.

    Splitting there keeps a chunk from cutting through the middle of an `if` body.
    """
    body = "\n".join(source_lines)
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []

    func = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if func is None:
        return []
    return [stmt.lineno - 1 for stmt in func.body]


def chunk_symbol(
    symbol: SymbolDef,
    file: SourceFile,
    repo_id: str,
    language: str = "python",
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[CodeChunk]:
    lines = file.content.splitlines()
    start = symbol.start_line - 1
    end = min(symbol.end_line, len(lines))
    body_lines = lines[start:end]
    if not body_lines:
        return []

    def make(content: str, part: int, total: int, first_line: int, last_line: int) -> CodeChunk:
        return CodeChunk(
            content=content,
            metadata=ChunkMetadata(
                qualified_name=symbol.qualified_name,
                file_path=symbol.file_path,
                start_line=first_line,
                end_line=last_line,
                language=language,
                parent_class=symbol.parent_class,
                repo_id=repo_id,
            ),
            part=part,
            total_parts=total,
        )

    if len(body_lines) <= max_lines:
        return [make("\n".join(body_lines), 0, 1, symbol.start_line, end)]

    # Dedent to column 0 so `ast.parse` accepts a method body lifted out of its class.
    indent = len(body_lines[0]) - len(body_lines[0].lstrip())
    dedented = [line[indent:] if len(line) >= indent else line for line in body_lines]
    boundaries = _statement_boundaries(dedented, symbol)

    header_end = boundaries[0] if boundaries else 1
    header = body_lines[:header_end]

    cut_points: list[int] = []
    current = header_end
    for boundary in boundaries:
        if boundary - current >= max_lines - len(header):
            cut_points.append(boundary)
            current = boundary
    segments: list[tuple[int, int]] = []
    previous = header_end
    for cut in [*cut_points, len(body_lines)]:
        if cut > previous:
            segments.append((previous, cut))
            previous = cut
    if not segments:
        segments = [(header_end, len(body_lines))]

    total = len(segments)
    chunks: list[CodeChunk] = []
    for index, (seg_start, seg_end) in enumerate(segments):
        content = "\n".join([*header, *body_lines[seg_start:seg_end]]) if index else "\n".join(
            body_lines[:seg_end]
        )
        chunks.append(
            make(
                content,
                index,
                total,
                symbol.start_line + (seg_start if index else 0),
                symbol.start_line + seg_end - 1,
            )
        )
    return chunks


def chunk_module(
    parsed: ParsedModule,
    file: SourceFile,
    repo_id: str,
    language: str = "python",
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for symbol in parsed.symbols:
        if symbol.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            continue
        chunks.extend(chunk_symbol(symbol, file, repo_id, language, max_lines))
    return chunks
