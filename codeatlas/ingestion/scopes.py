"""Lexical scopes and name bindings.

Plain mutable classes on purpose: this tree is built by an AST walk, mutated as the walk
descends, and consumed in-process by the resolver. Nothing here is serialised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ScopeKind = Literal["module", "class", "function", "comprehension"]


@dataclass
class Binding:
    """Base: a local name is bound to *something*."""

    name: str


@dataclass
class ImportBinding(Binding):
    """`import m as n` / `from m import s as n` / `from . import s`."""

    module: str
    symbol: str | None
    level: int = 0


@dataclass
class DefBinding(Binding):
    """A `def` or `class` statement in this scope."""

    qualified_name: str
    kind: Literal["function", "class"]


@dataclass
class ParamBinding(Binding):
    """A function parameter. `annotation` is the raw source text, resolved in pass 2."""

    annotation: str | None = None


@dataclass
class AssignBinding(Binding):
    """`x = Foo()` — `constructor` holds the raw callee name, resolved in pass 2."""

    constructor: str | None = None


@dataclass
class OpaqueBinding(Binding):
    """The name is bound, but to something we cannot type: `x = f()`, `for x in ...`.

    Distinct from "not found". An opaque binding correctly *shadows* an outer name, which
    is exactly what makes `import json` followed by a local `json = load()` resolve safely
    instead of silently pointing at the stdlib.
    """


@dataclass
class Scope:
    kind: ScopeKind
    qualified_name: str
    scope_id: str
    parent: Scope | None = None
    bindings: dict[str, Binding] = field(default_factory=dict)
    children: list[Scope] = field(default_factory=list)

    qn_prefix: str = ""
    """Prefix handed to symbols defined directly inside this scope."""

    class_bases: list[str] = field(default_factory=list)
    """Raw base expressions, for class scopes."""

    instance_attributes: dict[str, Binding] = field(default_factory=dict)
    """For class scopes: `self.x = Foo()` seen anywhere in the class body.

    `self._client.send()` is extremely common and is not a method lookup at all — without
    this map every such call site lands in `self_attr_not_found`, which is a pure recall
    loss on the calls most worth having.
    """

    module_name: str = ""
    file_path: str = ""

    def bind(self, binding: Binding) -> None:
        """Bind a name in this scope.

        A later binding overwrites an earlier one, except that we refuse to let an opaque
        binding erase a precise one — `def f(): ...` followed by `f = decorate(f)` should
        keep pointing at the definition rather than going dark.
        """
        existing = self.bindings.get(binding.name)
        if isinstance(binding, OpaqueBinding) and isinstance(existing, (DefBinding, ImportBinding)):
            return
        self.bindings[binding.name] = binding

    def add_child(self, child: Scope) -> Scope:
        child.parent = self
        self.children.append(child)
        return child

    def enclosing_class(self) -> Scope | None:
        """Nearest enclosing class scope, used to resolve `self.x` and `super()`."""
        cur: Scope | None = self
        while cur is not None:
            if cur.kind == "class":
                return cur
            cur = cur.parent
        return None

    def enclosing_module(self) -> Scope:
        cur: Scope = self
        while cur.parent is not None:
            cur = cur.parent
        return cur

    def lookup(self, name: str) -> tuple[Binding, Scope] | None:
        """Resolve `name` following Python's LEGB rule.

        The subtlety worth stating: a class body's namespace is **not** in the lookup chain
        of functions nested inside it. Inside a method, a bare `helper()` does not see a
        `helper` defined in the class body — it goes straight to module scope. Consulting
        the class scope here would mis-resolve a large number of method-internal calls.
        """
        cur: Scope | None = self
        is_start = True
        while cur is not None:
            skip = cur.kind == "class" and not is_start
            if not skip and name in cur.bindings:
                return cur.bindings[name], cur
            cur = cur.parent
            is_start = False
        return None


def iter_scopes(root: Scope):
    yield root
    for child in root.children:
        yield from iter_scopes(child)
