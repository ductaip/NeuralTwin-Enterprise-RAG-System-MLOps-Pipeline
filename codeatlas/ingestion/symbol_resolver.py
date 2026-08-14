"""Pass 2: resolve every recorded call site to a qualified name.

The governing rule is that we never guess. When the target cannot be established, the
edge is emitted with a low confidence and an explicit reason rather than being invented
or silently dropped:

* dropping it would cost **recall**, and for impact analysis a missing edge means a test
  that never runs, which means a bug reaching production;
* inventing it would cost **precision**, which here only means running a test that did
  not need to run.

Those costs are wildly asymmetric, so every edge we can name is written, carrying the
confidence that lets the *query* layer pick a threshold (see
`settings.CALL_EDGE_MIN_CONFIDENCE_*`).
"""

from __future__ import annotations

import builtins
import re
from collections import Counter
from dataclasses import dataclass, field

from loguru import logger

from codeatlas.ingestion.models import (
    CONF_ANNOTATED,
    CONF_DERIVED,
    CONF_DIRECT,
    CONF_INFERRED,
    CallEdge,
    InheritanceEdge,
    ParsedModule,
    RawCall,
    ResolutionStatus,
    SymbolDef,
    SymbolKind,
    TestEdge,
)
from codeatlas.ingestion.scopes import (
    AssignBinding,
    DefBinding,
    ImportBinding,
    OpaqueBinding,
    ParamBinding,
    Scope,
    iter_scopes,
)

BUILTIN_NAMES = frozenset(dir(builtins))
MAX_REEXPORT_HOPS = 3
MAX_MRO_DEPTH = 10

_SIMPLE_NAME_RE = re.compile(r"^[A-Za-z_][\w.]*$")
_WRAPPED_RE = re.compile(r"^(?:Optional|List|Sequence|Iterable|Set|Type|Awaitable|Coroutine|list|set|frozenset|tuple)\[(.+)\]$")


def _simplify_type_expression(expression: str) -> str:
    """Strip the wrappers a type annotation may carry down to a bare dotted name.

    `Optional[Foo]`, `list[Foo]`, `Foo | None` and `"Foo"` all reduce to `Foo`.
    """
    expression = expression.strip().strip("'\"")
    wrapped = _WRAPPED_RE.match(expression)
    if wrapped:
        expression = wrapped.group(1).split(",")[0].strip()
    if "|" in expression:
        candidates = [p.strip() for p in expression.split("|") if p.strip() not in ("None", "")]
        expression = candidates[0] if candidates else expression
    return expression.strip().strip("'\"")


@dataclass
class _Outcome:
    """Internal resolution result before it becomes a `CallEdge`."""

    target: str | None
    status: ResolutionStatus
    confidence: float
    reason: str


def _unresolved(reason: str) -> _Outcome:
    return _Outcome(None, ResolutionStatus.UNRESOLVED, 0.0, reason)


def _external(reason: str, target: str | None = None) -> _Outcome:
    return _Outcome(target, ResolutionStatus.EXTERNAL, CONF_DIRECT, reason)


@dataclass
class ResolutionReport:
    total_call_sites: int = 0
    internal: int = 0
    external: int = 0
    unresolved: int = 0
    reason_counts: Counter = field(default_factory=Counter)
    unresolved_reason_counts: Counter = field(default_factory=Counter)
    """Reasons for *failures* only.

    Kept separate rather than filtering `reason_counts` through a denylist of success
    labels: a denylist silently rots the moment a new success reason is added, and it did —
    it reported `self_attr_typed_own` as a failure pattern.
    """

    unresolved_samples: list[dict] = field(default_factory=list)

    @property
    def internal_call_sites(self) -> int:
        """Denominator for the resolve-rate gate.

        External calls are *successes* — `os.path.join` resolved correctly, to something
        outside the index. Folding them in either direction distorts the number, so the
        ≥80% gate is measured against internal + unresolved only. Fixed before any number
        was produced, deliberately.
        """
        return self.internal + self.unresolved

    @property
    def internal_resolve_rate(self) -> float:
        denom = self.internal_call_sites
        return self.internal / denom if denom else 0.0

    def top_patterns(self, n: int = 20) -> list[tuple[str, int]]:
        """Top failure patterns. A single dominant pattern is a morning's work, not a
        fundamental limit — which is the whole reason this is reported by pattern."""
        return self.unresolved_reason_counts.most_common(n)

    def to_dict(self) -> dict:
        return {
            "total_call_sites": self.total_call_sites,
            "internal": self.internal,
            "external": self.external,
            "unresolved": self.unresolved,
            "internal_call_sites": self.internal_call_sites,
            "internal_resolve_rate": round(self.internal_resolve_rate, 4),
            "top_failure_patterns": [
                {"reason": reason, "count": count} for reason, count in self.top_patterns(20)
            ],
            "unresolved_samples": self.unresolved_samples,
        }


@dataclass
class ResolutionResult:
    call_edges: list[CallEdge]
    inheritance_edges: list[InheritanceEdge]
    test_edges: list[TestEdge]
    report: ResolutionReport


class SymbolResolver:
    def __init__(self, modules: list[ParsedModule], max_unresolved_samples: int = 200):
        self.modules = modules
        self.max_unresolved_samples = max_unresolved_samples

        self.symbol_table: dict[str, SymbolDef] = {}
        self.scope_index: dict[str, Scope] = {}
        self.module_scopes: dict[str, Scope] = {}
        self.class_scopes: dict[str, Scope] = {}
        self.module_is_package: dict[str, bool] = {}
        self.modules_with_star_import: set[str] = set()

        for module in modules:
            self.module_is_package[module.module_name] = module.file_path.endswith("__init__.py")
            for symbol in module.symbols:
                # First definition wins. A genuine duplicate qualified name would mean two
                # different symbols collapsing into one graph node, so shout about it.
                if symbol.qualified_name in self.symbol_table:
                    existing = self.symbol_table[symbol.qualified_name]
                    if existing.file_path != symbol.file_path or existing.start_line != symbol.start_line:
                        logger.warning(
                            f"Duplicate qualified_name {symbol.qualified_name!r}: "
                            f"{existing.file_path}:{existing.start_line} vs "
                            f"{symbol.file_path}:{symbol.start_line}"
                        )
                    continue
                self.symbol_table[symbol.qualified_name] = symbol

            root = module.scope_root
            if isinstance(root, Scope):
                self.module_scopes[module.module_name] = root
                for scope in iter_scopes(root):
                    self.scope_index[scope.scope_id] = scope
                    if scope.kind == "class":
                        self.class_scopes[scope.qualified_name] = scope

            if any(imp.symbol == "*" for imp in module.imports):
                self.modules_with_star_import.add(module.module_name)

        self.indexed_modules = set(self.module_scopes)

    # ----------------------------------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------------------------------

    def resolve_all(self) -> ResolutionResult:
        report = ResolutionReport()
        call_edges: list[CallEdge] = []

        for module in self.modules:
            for raw in module.raw_calls:
                outcome = self.resolve_call(raw)
                report.total_call_sites += 1
                report.reason_counts[outcome.reason] += 1

                if outcome.status is ResolutionStatus.INTERNAL:
                    report.internal += 1
                elif outcome.status is ResolutionStatus.EXTERNAL:
                    report.external += 1
                else:
                    report.unresolved += 1
                    report.unresolved_reason_counts[outcome.reason] += 1
                    if len(report.unresolved_samples) < self.max_unresolved_samples:
                        report.unresolved_samples.append(
                            {
                                "file": raw.file_path,
                                "line": raw.line,
                                "caller": raw.caller_qn,
                                "callee_expr": ".".join(raw.dotted_parts) or "<dynamic>",
                                "reason": outcome.reason,
                            }
                        )

                # Only internal edges become graph edges: an edge to a symbol we never
                # indexed has no node to point at.
                if outcome.status is ResolutionStatus.INTERNAL and outcome.target:
                    call_edges.append(
                        CallEdge(
                            caller_qn=raw.caller_qn,
                            callee_qn=outcome.target,
                            line=raw.line,
                            confidence=outcome.confidence,
                            status=outcome.status,
                            reason=outcome.reason,
                        )
                    )

        return ResolutionResult(
            call_edges=call_edges,
            inheritance_edges=self.resolve_inheritance(),
            test_edges=self.derive_test_edges(call_edges),
            report=report,
        )

    # ----------------------------------------------------------------------------------
    # Call resolution
    # ----------------------------------------------------------------------------------

    def resolve_call(self, raw: RawCall) -> _Outcome:
        scope = self.scope_index.get(raw.scope_id)
        if scope is None:
            return _unresolved("scope_missing")

        if not raw.dotted_parts:
            # `d[key]()`, `f()()`, `(a or b)()` — a real call site whose target cannot be
            # named without dataflow analysis. Counted, never guessed.
            return _unresolved("dynamic_callee")

        if raw.is_super_call:
            return self.resolve_super(scope, raw.dotted_parts[0])

        head, *rest = raw.dotted_parts

        if head in ("self", "cls") and rest:
            return self.resolve_self_attribute(scope, rest)

        found = scope.lookup(head)
        if found is None:
            if head in BUILTIN_NAMES:
                return _external("builtin", f"builtins.{head}")
            if scope.module_name in self.modules_with_star_import:
                return _unresolved("star_import_shadow")
            return _unresolved("name_not_found")

        binding, _owner = found
        return self.resolve_binding(binding, rest, scope)

    def resolve_binding(self, binding, rest: list[str], scope: Scope) -> _Outcome:
        if isinstance(binding, DefBinding):
            return self._resolve_def_binding(binding, rest)

        if isinstance(binding, ImportBinding):
            return self._resolve_import_binding(binding, rest, scope)

        if isinstance(binding, ParamBinding):
            if not binding.annotation:
                return _unresolved("param_untyped")
            class_qn = self._resolve_type_expression(binding.annotation, scope)
            if class_qn is None:
                # An annotation naming a third-party type is a *resolved* answer that
                # happens to point outside the index — same bucket as `os.path.join`.
                if self._expression_is_external(binding.annotation, scope):
                    return _external("annotated_external")
                return _unresolved("annotation_unresolvable")
            if not rest:
                return _unresolved("call_on_annotated_value")
            return self._attribute_on_class(class_qn, rest[0], base_confidence=CONF_ANNOTATED,
                                            reason_prefix="annotated")

        if isinstance(binding, AssignBinding):
            if not binding.constructor:
                return _unresolved("assign_untyped")
            class_qn = self._resolve_type_expression(binding.constructor, scope)
            if class_qn is None:
                # `engine = build_engine()` where `build_engine` is annotated `-> Engine`.
                # The author wrote the return type down; reading it back is not guessing.
                inferred = self._infer_return_type(binding.constructor, scope)
                if inferred is not None and rest:
                    return self._attribute_on_class(
                        inferred, rest[0], base_confidence=CONF_INFERRED,
                        reason_prefix="inferred_return",
                    )
                # Separate the two ways this fails, because they need different fixes and
                # the reason histogram is what tells us which one dominates:
                #   `x = make_thing()` -> we would need return-type inference
                #   `x = whatever()`   -> we could not even resolve the callee
                if self._expression_is_external(binding.constructor, scope):
                    return _external("constructor_external")
                if self._resolves_to_function(binding.constructor, scope):
                    return _unresolved("assign_from_function_return")
                if self._value_is_externally_derived(binding.constructor, scope):
                    # `client = TestClient(app)` then `response = client.get(...)` then
                    # `response.json()`. Every link in that chain is outside the index, so
                    # the call is external for the same reason `os.path.join` is — a known
                    # unknown, not a resolver failure.
                    return _external("derived_from_external")
                if self._is_method_call_expression(binding.constructor, scope):
                    # `response = client.get(...)` then `response.json()`. Not a resolution
                    # failure so much as a missing capability: we would need return-type
                    # inference. Labelled separately so the histogram does not read as if
                    # the scope chain were broken.
                    return _unresolved("assign_from_method_return")
                return _unresolved("constructor_unresolvable")
            if not rest:
                return _unresolved("call_on_instance_value")
            return self._attribute_on_class(class_qn, rest[0], base_confidence=CONF_INFERRED,
                                            reason_prefix="inferred_instance")

        if isinstance(binding, OpaqueBinding):
            # The name is bound to something we cannot type, but it *is* bound — which is
            # what stops `json = load()` inside a function from resolving to the stdlib.
            return _unresolved("opaque_binding")

        return _unresolved("unknown_binding_kind")

    def _resolve_def_binding(self, binding: DefBinding, rest: list[str]) -> _Outcome:
        if not rest:
            if binding.kind == "class":
                return self._instantiation_target(binding.qualified_name)
            return _Outcome(binding.qualified_name, ResolutionStatus.INTERNAL, CONF_DIRECT, "direct_local")

        if binding.kind == "class":
            return self._attribute_on_class(binding.qualified_name, rest[0],
                                            base_confidence=CONF_DIRECT, reason_prefix="class_attr")

        nested = f"{binding.qualified_name}.{rest[0]}"
        if nested in self.symbol_table:
            return _Outcome(nested, ResolutionStatus.INTERNAL, CONF_DIRECT, "direct_nested")
        return _unresolved("attr_on_function")

    def _resolve_import_binding(self, binding: ImportBinding, rest: list[str], scope: Scope) -> _Outcome:
        module = self._absolute_module(binding, scope)
        if module is None:
            return _unresolved("relative_import_beyond_root")

        if binding.symbol is None:
            # `import a.b as z` — the binding is a module; the first remaining part is the
            # symbol inside it.
            if not rest:
                return _unresolved("call_on_module")
            return self._resolve_module_symbol(module, rest[0], rest[1:])

        return self._resolve_module_symbol(module, binding.symbol, rest)

    def _resolve_module_symbol(
        self, module: str, symbol: str, rest: list[str], hops: int = 0, as_type: bool = False
    ) -> _Outcome:
        """Resolve `module.symbol`, following re-export chains through `__init__.py`.

        `as_type=True` is used when the caller wants the *class* rather than the thing a
        call on it would reach: `app = FastAPI()` needs `fastapi.applications.FastAPI`, not
        that class's `__init__`.
        """
        dotted = f"{module}.{symbol}"

        # `import a.b` then `a.b.c.f()` — the dotted path may itself name a deeper module.
        deeper = self._longest_indexed_module(dotted)
        if deeper == dotted and rest:
            return self._resolve_module_symbol(dotted, rest[0], rest[1:], hops, as_type)

        target = self.symbol_table.get(dotted)
        if target is not None:
            confidence = CONF_DIRECT if hops == 0 else CONF_DERIVED
            reason = "via_import" if hops == 0 else "via_reexport"
            if rest:
                if target.kind is SymbolKind.CLASS:
                    return self._attribute_on_class(dotted, rest[0], base_confidence=confidence,
                                                    reason_prefix="imported_class_attr")
                nested = f"{dotted}.{rest[0]}"
                if nested in self.symbol_table:
                    return _Outcome(nested, ResolutionStatus.INTERNAL, confidence, reason)
                return _unresolved("attr_on_imported_symbol")
            if target.kind is SymbolKind.CLASS and not as_type:
                return self._instantiation_target(dotted, confidence)
            return _Outcome(dotted, ResolutionStatus.INTERNAL, confidence, reason)

        if module not in self.indexed_modules:
            return _external("external_module", dotted)

        module_scope = self.module_scopes.get(module)
        if module_scope is not None and hops < MAX_REEXPORT_HOPS:
            binding = module_scope.bindings.get(symbol)
            if isinstance(binding, ImportBinding):
                next_module = self._absolute_module(binding, module_scope)
                if next_module is not None:
                    next_symbol = binding.symbol
                    if next_symbol is None:
                        # Re-exported module rather than symbol.
                        if not rest:
                            return _unresolved("call_on_module")
                        return self._resolve_module_symbol(
                            next_module, rest[0], rest[1:], hops + 1, as_type
                        )
                    return self._resolve_module_symbol(
                        next_module, next_symbol, rest, hops + 1, as_type
                    )
            if isinstance(binding, DefBinding):
                if as_type and binding.kind == "class" and not rest:
                    return _Outcome(
                        binding.qualified_name, ResolutionStatus.INTERNAL, CONF_DERIVED, "via_reexport"
                    )
                return self._resolve_def_binding(binding, rest)

        return _unresolved("symbol_not_in_module")

    def _instantiation_target(self, class_qn: str, confidence: float = CONF_DIRECT) -> _Outcome:
        """`MyClass()` — point the edge at `__init__` when there is one.

        When no `__init__` exists anywhere in the MRO the edge is aimed at the class node
        itself. That is a deliberate deviation from the strict `Function -> Function` shape
        in the spec: dropping instantiation edges would blind impact analysis to every
        caller that constructs the class, which is precisely the recall hole we agreed to
        avoid.
        """
        init = self._lookup_in_mro(class_qn, "__init__")
        if init is not None:
            target, depth = init
            conf = confidence if depth == 0 else min(confidence, CONF_ANNOTATED)
            return _Outcome(target, ResolutionStatus.INTERNAL, conf, "instantiation_init")
        return _Outcome(class_qn, ResolutionStatus.INTERNAL, confidence, "instantiation_class")

    # ----------------------------------------------------------------------------------
    # self / super / MRO
    # ----------------------------------------------------------------------------------

    def resolve_self_attribute(self, scope: Scope, attr_parts: list[str]) -> _Outcome:
        cls_scope = scope.enclosing_class()
        if cls_scope is None:
            return _unresolved("self_outside_class")

        attr, *rest = attr_parts
        found = self._lookup_in_mro(cls_scope.qualified_name, attr)

        if found is not None and not rest:
            target, depth = found
            if depth == 0:
                return _Outcome(target, ResolutionStatus.INTERNAL, CONF_DIRECT, "self_attr_own")
            confidence = CONF_ANNOTATED if depth == 1 else CONF_DERIVED
            return _Outcome(target, ResolutionStatus.INTERNAL, confidence, "self_attr_inherited")

        # `self._client.send()` — `_client` is data, not a method. Type it from wherever the
        # class assigned it and continue the lookup on that type.
        attr_binding = self._lookup_instance_attribute(cls_scope, attr)
        if attr_binding is not None and rest:
            return self._resolve_typed_attribute(
                attr_binding, rest[0], cls_scope, reason_prefix="self_attr_typed"
            )

        if found is not None and rest:
            return _unresolved("self_method_result_attr")

        if attr_binding is not None:
            return _unresolved("self_attr_is_data")

        if self._has_external_base(cls_scope):
            # Inherited from something outside the index (BaseModel, ABC, ...).
            return _external("self_attr_external_base")
        return _unresolved("self_attr_not_found")

    def _lookup_instance_attribute(self, cls_scope: Scope, attr: str):
        """Find `self.<attr>` on the class or any indexed ancestor."""
        seen: set[str] = set()
        frontier = [cls_scope.qualified_name]

        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)

            scope = self.class_scopes.get(current)
            if scope is not None:
                binding = scope.instance_attributes.get(attr) or scope.bindings.get(attr)
                if binding is not None and not isinstance(binding, OpaqueBinding):
                    return binding

            symbol = self.symbol_table.get(current)
            if symbol is None or not symbol.bases:
                continue
            base_scope = self.module_scopes.get(symbol.module_name)
            if base_scope is None:
                continue
            for base_expr in symbol.bases:
                base_qn = self._resolve_type_expression(base_expr, base_scope)
                if base_qn is not None:
                    frontier.append(base_qn)

        return None

    def _resolve_typed_attribute(
        self, binding, attr: str, scope: Scope, reason_prefix: str
    ) -> _Outcome:
        """Given a binding that carries a type, resolve `<value>.<attr>()`."""
        if isinstance(binding, ParamBinding):
            expression, confidence = binding.annotation, CONF_ANNOTATED
        elif isinstance(binding, AssignBinding):
            expression, confidence = binding.constructor, CONF_INFERRED
        else:
            return _unresolved(f"{reason_prefix}_untyped")

        if not expression:
            return _unresolved(f"{reason_prefix}_untyped")

        lookup_scope = self.module_scopes.get(scope.module_name, scope)
        class_qn = self._resolve_type_expression(expression, lookup_scope)
        if class_qn is None:
            if self._expression_is_external(expression, lookup_scope):
                return _external(f"{reason_prefix}_external")
            return _unresolved(f"{reason_prefix}_unresolvable")

        return self._attribute_on_class(
            class_qn, attr, base_confidence=confidence, reason_prefix=reason_prefix
        )

    def resolve_super(self, scope: Scope, method: str) -> _Outcome:
        cls_scope = scope.enclosing_class()
        if cls_scope is None:
            return _unresolved("super_outside_class")
        if not cls_scope.class_bases:
            # Implicitly `object`.
            return _external("super_object")

        parent_scope = cls_scope.parent or cls_scope
        multiple = len(cls_scope.class_bases) > 1

        for base_expr in cls_scope.class_bases:
            base_qn = self._resolve_type_expression(base_expr, parent_scope)
            if base_qn is None:
                continue
            found = self._lookup_in_mro(base_qn, method)
            if found is not None:
                target, _depth = found
                # Under multiple inheritance the real MRO is C3-linearised; walking bases in
                # declaration order agrees with C3 for the common cases and can diverge for
                # diamonds, so the confidence drops rather than pretending otherwise.
                confidence = CONF_DERIVED if multiple else CONF_ANNOTATED
                reason = "super_multi_base" if multiple else "super_single_base"
                return _Outcome(target, ResolutionStatus.INTERNAL, confidence, reason)

        if self._has_external_base(cls_scope):
            return _external("super_external_base")
        return _unresolved("super_method_not_found")

    def _has_external_base(self, cls_scope: Scope) -> bool:
        parent_scope = cls_scope.parent or cls_scope
        for base_expr in cls_scope.class_bases:
            if self._resolve_type_expression(base_expr, parent_scope) is None:
                return True
        return False

    def _lookup_in_mro(self, class_qn: str, attr: str) -> tuple[str, int] | None:
        """Breadth-first walk of the inheritance graph. Returns (target_qn, depth)."""
        seen: set[str] = set()
        frontier = [(class_qn, 0)]

        while frontier:
            current, depth = frontier.pop(0)
            if current in seen or depth > MAX_MRO_DEPTH:
                continue
            seen.add(current)

            candidate = f"{current}.{attr}"
            if candidate in self.symbol_table:
                return candidate, depth

            symbol = self.symbol_table.get(current)
            if symbol is None or not symbol.bases:
                continue
            base_scope = self.module_scopes.get(symbol.module_name)
            if base_scope is None:
                continue
            for base_expr in symbol.bases:
                base_qn = self._resolve_type_expression(base_expr, base_scope)
                if base_qn is not None:
                    frontier.append((base_qn, depth + 1))

        return None

    def _attribute_on_class(
        self, class_qn: str, attr: str, base_confidence: float, reason_prefix: str
    ) -> _Outcome:
        found = self._lookup_in_mro(class_qn, attr)
        if found is None:
            return _unresolved(f"{reason_prefix}_not_found")
        target, depth = found
        confidence = base_confidence if depth == 0 else min(base_confidence, CONF_DERIVED)
        suffix = "own" if depth == 0 else "inherited"
        return _Outcome(target, ResolutionStatus.INTERNAL, confidence, f"{reason_prefix}_{suffix}")

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------

    def _absolute_module(self, binding: ImportBinding, scope: Scope) -> str | None:
        """Turn a possibly-relative import into an absolute dotted module path."""
        if binding.level == 0:
            return binding.module

        module_name = scope.module_name
        is_package = self.module_is_package.get(module_name, False)
        parts = module_name.split(".")
        package_parts = parts if is_package else parts[:-1]

        drop = binding.level - 1
        if drop > len(package_parts):
            return None
        target_parts = package_parts[: len(package_parts) - drop] if drop else package_parts

        if binding.module:
            target_parts = [*target_parts, *binding.module.split(".")]
        return ".".join(target_parts) if target_parts else None

    def _expression_is_external(self, expression: str, scope: Scope) -> bool:
        """True when a type expression names something imported from outside the index.

        `def f(client: OpenAI)` cannot be resolved to a node, but it is not a resolver
        failure — the type is simply not part of this repository. Keeping it out of the
        unresolved bucket is the same rule that keeps `os.path.join` out of it.
        """
        head, *rest = _simplify_type_expression(expression).split(".")
        if head in BUILTIN_NAMES:
            return True
        found = scope.lookup(head)
        if found is None:
            return False
        binding, _owner = found
        if not isinstance(binding, ImportBinding):
            return False
        module = self._absolute_module(binding, scope)
        if module is None:
            return False
        if self._longest_indexed_module(module) is None:
            return True

        # The first hop landing inside the index proves nothing: `fastapi.Request` is a
        # re-export of `starlette.requests.Request`. Follow the chain to where the symbol
        # actually lives before deciding which bucket this belongs in.
        if binding.symbol is None:
            if not rest:
                return False
            outcome = self._resolve_module_symbol(module, rest[0], rest[1:], as_type=True)
        else:
            outcome = self._resolve_module_symbol(module, binding.symbol, rest, as_type=True)
        return outcome.status is ResolutionStatus.EXTERNAL

    def _infer_return_type(self, expression: str, scope: Scope) -> str | None:
        """Class returned by `expression`, taken from the callee's return annotation.

        Deliberately restricted to expressions rooted in a `def`/`import` binding. Following
        an assignment chain here would recurse straight back into this method through the
        `AssignBinding` branch.
        """
        head, *rest = _simplify_type_expression(expression).split(".")
        found = scope.lookup(head)
        if found is None:
            return None

        binding, _owner = found
        if not isinstance(binding, (DefBinding, ImportBinding)):
            return None

        outcome = self.resolve_binding(binding, rest, scope)
        if outcome.status is not ResolutionStatus.INTERNAL or not outcome.target:
            return None

        symbol = self.symbol_table.get(outcome.target)
        if symbol is None or not symbol.returns:
            return None

        declaring_scope = self.module_scopes.get(symbol.module_name, scope)
        return self._resolve_type_expression(symbol.returns, declaring_scope)

    def _value_is_externally_derived(self, expression: str, scope: Scope, depth: int = 0) -> bool:
        """True when `expression` is a call on a value that ultimately came from outside.

        Walks the assignment chain — `response` <- `client.get` <- `client` <- `TestClient` —
        rather than only inspecting the immediate name.
        """
        if depth > MAX_REEXPORT_HOPS:
            return False

        head, *rest = _simplify_type_expression(expression).split(".")
        if not rest:
            return False

        found = scope.lookup(head)
        if found is None:
            return False

        binding, _owner = found
        if isinstance(binding, ParamBinding) and binding.annotation:
            return self._expression_is_external(binding.annotation, scope)
        if isinstance(binding, AssignBinding) and binding.constructor:
            if self._expression_is_external(binding.constructor, scope):
                return True
            return self._value_is_externally_derived(binding.constructor, scope, depth + 1)
        return False

    def _is_method_call_expression(self, expression: str, scope: Scope) -> bool:
        """True for `something.method` where `something` is a local value, not a module."""
        head, *rest = _simplify_type_expression(expression).split(".")
        if not rest:
            return False
        found = scope.lookup(head)
        if found is None:
            return False
        binding, _owner = found
        return isinstance(binding, (AssignBinding, ParamBinding, OpaqueBinding))

    def _resolves_to_function(self, expression: str, scope: Scope) -> bool:
        """True when `expression` names a function we know about (so a call returns an
        unknown type) rather than a name we failed to resolve at all."""
        head = expression.split(".")[0]
        found = scope.lookup(head)
        if found is None:
            return False
        binding, _owner = found
        if isinstance(binding, DefBinding):
            return binding.kind == "function"
        if isinstance(binding, ImportBinding):
            module = self._absolute_module(binding, scope)
            if module is None or binding.symbol is None:
                return False
            target = self.symbol_table.get(f"{module}.{binding.symbol}")
            return target is not None and target.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)
        return False

    def _longest_indexed_module(self, dotted: str) -> str | None:
        parts = dotted.split(".")
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in self.indexed_modules:
                return candidate
        return None

    def _resolve_type_expression(self, expression: str, scope: Scope) -> str | None:
        """Resolve a type/base expression such as `Foo`, `mod.Foo`, `Optional[Foo]` to a qn."""
        expression = _simplify_type_expression(expression)

        if not _SIMPLE_NAME_RE.match(expression):
            return None

        head, *rest = expression.split(".")
        found = scope.lookup(head)
        if found is None:
            return None

        binding, _owner = found
        if isinstance(binding, DefBinding):
            if binding.kind != "class" and not rest:
                return None
            qn = binding.qualified_name
            for part in rest:
                qn = f"{qn}.{part}"
            return qn if qn in self.symbol_table else None

        if isinstance(binding, ImportBinding):
            module = self._absolute_module(binding, scope)
            if module is None:
                return None
            # Go through the same machinery as call resolution rather than a simpler
            # lookup of our own. A weaker second path here is what made every
            # `app = FastAPI()` fail: `fastapi.FastAPI` is a re-export, and the class
            # actually lives at `fastapi.applications.FastAPI`.
            if binding.symbol is None:
                if not rest:
                    return None
                outcome = self._resolve_module_symbol(module, rest[0], rest[1:], as_type=True)
            else:
                outcome = self._resolve_module_symbol(module, binding.symbol, rest, as_type=True)

            if outcome.status is not ResolutionStatus.INTERNAL or not outcome.target:
                return None
            symbol = self.symbol_table.get(outcome.target)
            return outcome.target if symbol and symbol.kind is SymbolKind.CLASS else None

        return None

    # ----------------------------------------------------------------------------------
    # Derived edges
    # ----------------------------------------------------------------------------------

    def resolve_inheritance(self) -> list[InheritanceEdge]:
        edges: list[InheritanceEdge] = []
        for symbol in self.symbol_table.values():
            if symbol.kind is not SymbolKind.CLASS or not symbol.bases:
                continue
            scope = self.module_scopes.get(symbol.module_name)
            if scope is None:
                continue
            for base_expr in symbol.bases:
                base_qn = self._resolve_type_expression(base_expr, scope)
                if base_qn is None:
                    edges.append(
                        InheritanceEdge(
                            child_qn=symbol.qualified_name,
                            parent_qn=base_expr,
                            status=ResolutionStatus.EXTERNAL,
                            confidence=CONF_DIRECT,
                        )
                    )
                else:
                    edges.append(
                        InheritanceEdge(
                            child_qn=symbol.qualified_name,
                            parent_qn=base_qn,
                            status=ResolutionStatus.INTERNAL,
                            confidence=CONF_DIRECT,
                        )
                    )
        return edges

    def derive_test_edges(self, call_edges: list[CallEdge]) -> list[TestEdge]:
        """A test `TESTS` whatever it calls directly.

        Cypher [3] then walks `CALLS*0..3` outward from there, so a direct-call edge is the
        right granularity — inferring more here would double-count what the traversal does.
        """
        seen: set[tuple[str, str]] = set()
        edges: list[TestEdge] = []

        for edge in call_edges:
            caller = self.symbol_table.get(edge.caller_qn)
            callee = self.symbol_table.get(edge.callee_qn)
            if caller is None or not caller.is_test:
                continue
            if callee is None or callee.is_test:
                continue
            key = (edge.caller_qn, edge.callee_qn)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                TestEdge(
                    test_qn=edge.caller_qn,
                    target_qn=edge.callee_qn,
                    confidence=edge.confidence,
                )
            )
        return edges
