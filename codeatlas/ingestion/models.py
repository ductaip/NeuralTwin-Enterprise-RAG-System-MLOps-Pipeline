"""Data models for the ingestion layer.

Everything that crosses a module boundary is a pydantic v2 model so the shapes are
validated once, at the seam. The scope tree built while walking the AST is a plain
class instead — it is recursive, mutated in place, and never leaves the parser.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------------------


CONF_DIRECT = 1.0
"""Name resolved through an unambiguous binding to a definition present in the index."""

CONF_ANNOTATED = 0.9
"""Resolved via an explicit type annotation, or one inheritance hop under single
inheritance. The programmer wrote the type down; we are only reading it back."""

CONF_DERIVED = 0.7
"""Resolved through a re-export chain, a multi-hop MRO walk, or `super()` under multiple
inheritance. Each of those involves an approximation we can name."""

CONF_INFERRED = 0.5
"""Best-effort instance typing: `x = Foo()` then `x.bar()`. Correct in the common case,
defeated by reassignment and by any indirection."""


class ResolutionStatus(StrEnum):
    INTERNAL = "internal"
    """Resolved to a symbol defined inside the indexed repository."""

    EXTERNAL = "external"
    """Correctly resolved to something outside the indexed file set (stdlib, third party).

    This is a *success*, not a failure. Counting external calls as unresolved would sink
    the resolve rate for reasons that have nothing to do with resolver quality; counting
    them as resolved would inflate it. They get their own bucket.
    """

    UNRESOLVED = "unresolved"
    """We could not determine the target. Never guessed at."""


class SymbolKind(StrEnum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


# --------------------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------------------


class SourceFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    """Repository-relative POSIX path, e.g. `fastapi/routing.py`."""

    module_name: str
    """Dotted module path, e.g. `fastapi.routing`."""

    content: str
    sha: str
    loc: int


# --------------------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------------------


class SymbolDef(BaseModel):
    qualified_name: str
    """Unique key. Follows `__qualname__` semantics, so a function nested inside another
    function carries the `<locals>` marker: `pkg.mod.outer.<locals>.inner`.

    The marker is ugly in Cypher but mandatory: `qualified_name` is the MERGE key, and two
    distinct `inner` functions collapsing into one node is a silent corruption — no error,
    just a graph that answers impact queries wrongly.
    """

    name: str
    kind: SymbolKind
    file_path: str
    module_name: str
    start_line: int
    end_line: int
    is_async: bool = False
    is_public: bool = True
    is_test: bool = False
    docstring: str | None = None
    signature: str | None = None
    returns: str | None = None
    """Raw return annotation, used to type `x = make_thing()` at the call site."""

    decorators: list[str] = Field(default_factory=list)
    parent_class: str | None = None
    """Qualified name of the enclosing class, for methods."""

    bases: list[str] = Field(default_factory=list)
    """Raw base-class expressions as written, e.g. `["BaseModel", "abc.ABC"]`."""

    complexity: int = 1


class ImportRecord(BaseModel):
    """One `import` / `from ... import ...` binding, as written."""

    module: str
    """Dotted module being imported from. Empty for a bare relative `from . import x`."""

    symbol: str | None = None
    """The imported name for `from m import s`; None for `import m`."""

    alias: str
    """The local name the binding is reachable under."""

    level: int = 0
    """Relative-import depth: 0 absolute, 1 `.`, 2 `..`, and so on."""

    line: int = 0
    file_path: str = ""


class RawCall(BaseModel):
    """An unresolved call site: what was written, and where."""

    caller_qn: str
    """Qualified name of the enclosing function/method. Module-level calls use the module."""

    scope_id: str
    """Identity of the lexical scope the call appears in, for scope-chain lookup."""

    file_path: str
    line: int

    dotted_parts: list[str]
    """The callee expression flattened: `foo()` -> ["foo"], `z.a.b()` -> ["z","a","b"],
    `self.m()` -> ["self","m"]."""

    is_super_call: bool = False


class CallEdge(BaseModel):
    caller_qn: str
    callee_qn: str
    line: int
    confidence: float
    status: ResolutionStatus
    reason: str
    """Machine-readable label for *why* this confidence, e.g. `via_import`,
    `self_attr_inherited`, `name_not_found`. Aggregated into the unresolved report so a
    dominant failure pattern is visible as one line rather than a thousand."""


class InheritanceEdge(BaseModel):
    child_qn: str
    parent_qn: str
    status: ResolutionStatus
    confidence: float


class TestEdge(BaseModel):
    test_qn: str
    target_qn: str
    confidence: float


# --------------------------------------------------------------------------------------
# Parse result
# --------------------------------------------------------------------------------------


class ParsedModule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_name: str
    file_path: str
    symbols: list[SymbolDef] = Field(default_factory=list)
    imports: list[ImportRecord] = Field(default_factory=list)
    raw_calls: list[RawCall] = Field(default_factory=list)
    scope_root: object = None
    """The `Scope` tree produced while walking the AST. Typed as `object` because `Scope`
    is a plain mutable class that deliberately stays out of pydantic."""

    parse_error: str | None = None


class ChunkMetadata(BaseModel):
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    parent_class: str | None = None
    repo_id: str


class CodeChunk(BaseModel):
    content: str
    metadata: ChunkMetadata
    part: int = 0
    """Index of this chunk within its function, for functions split across chunks."""

    total_parts: int = 1


# --------------------------------------------------------------------------------------
# Parser protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class Parser(Protocol):
    """Contract every language front-end implements.

    Adding tree-sitter for another language means writing one class satisfying this
    protocol. `graph_builder` and `symbol_resolver` consume `ParsedModule` and never
    touch a syntax tree, so neither needs to change.
    """

    language: str

    def can_parse(self, path: str) -> bool: ...

    def parse(self, file: SourceFile) -> ParsedModule: ...
