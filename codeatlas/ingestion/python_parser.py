"""Pass 1: parse one Python file into symbols, scopes, bindings and raw call sites.

Standard-library `ast` only. No resolution happens here — a call site in this file may
target a definition in another file that has not been parsed yet, so every callee is
recorded verbatim and handed to `symbol_resolver` for pass 2.
"""

from __future__ import annotations

import ast
import re
from itertools import count

from loguru import logger

from codeatlas.ingestion.models import (
    ImportRecord,
    ParsedModule,
    RawCall,
    SourceFile,
    SymbolDef,
    SymbolKind,
)
from codeatlas.ingestion.scopes import (
    AssignBinding,
    DefBinding,
    ImportBinding,
    OpaqueBinding,
    ParamBinding,
    Scope,
)

TEST_FILE_RE = re.compile(r"(^|/)(test_[^/]+|[^/]+_test)\.py$")

_DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.IfExp,
    ast.Assert,
    ast.Match,
)


def _is_test_file(path: str) -> bool:
    return bool(TEST_FILE_RE.search(path))


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - malformed nodes
        return None


def _cyclomatic_complexity(node: ast.AST) -> int:
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, _DECISION_NODES):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            complexity += 1 + len(child.ifs)
    return complexity


def _literal_type_name(node: ast.expr) -> str | None:
    """Builtin type a literal expression evaluates to, or None if it is not a literal."""
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Constant):
        return {
            str: "str",
            bytes: "bytes",
            bool: "bool",
            int: "int",
            float: "float",
        }.get(type(node.value))
    return None


def _is_public(name: str) -> bool:
    """Dunders count as public: they are invoked implicitly, so they are never dead code."""
    if name.startswith("__") and name.endswith("__"):
        return True
    return not name.startswith("_")


def flatten_callee(func: ast.expr) -> tuple[list[str], bool]:
    """Flatten a callee expression into dotted parts.

    `foo()` -> (["foo"], True); `a.b.c()` -> (["a","b","c"], True).
    Anything rooted in something other than a plain name — `d[k]()`, `f()()`, `(a or b)()` —
    returns (…, False). Those are still recorded as call sites so they stay in the
    denominator; hiding them would flatter the resolve rate.
    """
    parts: list[str] = []
    cur: ast.expr = func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts, True
    parts.reverse()
    return parts, False


def _is_super_call(func: ast.expr) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
        and func.value.func.id == "super"
    )


class PythonParser:
    """Implements the `Parser` protocol for Python via the stdlib `ast` module."""

    language = "python"

    def can_parse(self, path: str) -> bool:
        return path.endswith(".py")

    def parse(self, file: SourceFile) -> ParsedModule:
        try:
            tree = ast.parse(file.content, filename=file.path)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file.path}: {e}")
            return ParsedModule(
                module_name=file.module_name,
                file_path=file.path,
                parse_error=f"{type(e).__name__}: {e}",
            )

        visitor = _ModuleVisitor(file)
        visitor.run(tree)

        return ParsedModule(
            module_name=file.module_name,
            file_path=file.path,
            symbols=visitor.symbols,
            imports=visitor.imports,
            raw_calls=visitor.raw_calls,
            scope_root=visitor.module_scope,
        )


class _ModuleVisitor:
    def __init__(self, file: SourceFile):
        self.file = file
        self.symbols: list[SymbolDef] = []
        self.imports: list[ImportRecord] = []
        self.raw_calls: list[RawCall] = []
        self._scope_ids = count()
        self.is_test_file = _is_test_file(file.path)

        self.module_scope = Scope(
            kind="module",
            qualified_name=file.module_name,
            scope_id=self._new_scope_id(),
            qn_prefix=file.module_name,
            module_name=file.module_name,
            file_path=file.path,
        )
        self._scope_stack: list[Scope] = [self.module_scope]
        # Qualified name of the innermost *function*; call sites are attributed to it.
        self._caller_stack: list[str] = [file.module_name]

    def _new_scope_id(self) -> str:
        return f"{self.file.path}#{next(self._scope_ids)}"

    @property
    def scope(self) -> Scope:
        return self._scope_stack[-1]

    @property
    def caller_qn(self) -> str:
        return self._caller_stack[-1]

    def run(self, tree: ast.Module) -> None:
        self.symbols.append(
            SymbolDef(
                qualified_name=self.file.module_name,
                name=self.file.module_name.rsplit(".", 1)[-1],
                kind=SymbolKind.MODULE,
                file_path=self.file.path,
                module_name=self.file.module_name,
                start_line=1,
                end_line=self.file.loc,
                docstring=ast.get_docstring(tree),
            )
        )
        self._visit_body(tree.body)

    # -- dispatch ----------------------------------------------------------------------

    def _visit_body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self._visit(stmt)

    def _visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._handle_import(node)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._handle_function(node)
            return
        if isinstance(node, ast.ClassDef):
            self._handle_class(node)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            self._handle_assign(node)
            # fall through so calls on the right-hand side are still recorded
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self._bind_target_opaque(node.target)
        if isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
            for item in node.items:
                if item.optional_vars is not None:
                    self._bind_target_opaque(item.optional_vars)
        if isinstance(node, ast.ExceptHandler) and node.name:
            self.scope.bind(OpaqueBinding(name=node.name))
        if isinstance(node, ast.Call):
            self._handle_call(node)

        for child in ast.iter_child_nodes(node):
            self._visit(child)

    # -- imports -----------------------------------------------------------------------

    def _handle_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import a.b.c` binds the *top* name `a`; `import a.b.c as z` binds `z`
                # to the full dotted module. Conflating the two mis-resolves `a.b.c.f()`.
                local = alias.asname or alias.name.split(".")[0]
                module = alias.name if alias.asname else alias.name.split(".")[0]
                record = ImportRecord(
                    module=module,
                    symbol=None,
                    alias=local,
                    level=0,
                    line=node.lineno,
                    file_path=self.file.path,
                )
                self.imports.append(record)
                self.scope.bind(
                    ImportBinding(name=local, module=module, symbol=None, level=0)
                )
            return

        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                # Star imports make the scope unknowable. Record it, bind nothing, and let
                # unresolved names in this module carry the `star_import` reason.
                self.imports.append(
                    ImportRecord(
                        module=module,
                        symbol="*",
                        alias="*",
                        level=node.level,
                        line=node.lineno,
                        file_path=self.file.path,
                    )
                )
                continue
            local = alias.asname or alias.name
            record = ImportRecord(
                module=module,
                symbol=alias.name,
                alias=local,
                level=node.level,
                line=node.lineno,
                file_path=self.file.path,
            )
            self.imports.append(record)
            self.scope.bind(
                ImportBinding(name=local, module=module, symbol=alias.name, level=node.level)
            )

    # -- definitions -------------------------------------------------------------------

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qn = f"{self.scope.qn_prefix}.{node.name}" if self.scope.qn_prefix else node.name
        enclosing_class = self.scope if self.scope.kind == "class" else None
        decorators = [d for d in (_unparse(d) for d in node.decorator_list) if d]

        self.scope.bind(DefBinding(name=node.name, qualified_name=qn, kind="function"))

        self.symbols.append(
            SymbolDef(
                qualified_name=qn,
                name=node.name,
                kind=SymbolKind.METHOD if enclosing_class else SymbolKind.FUNCTION,
                file_path=self.file.path,
                module_name=self.file.module_name,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_public=_is_public(node.name),
                is_test=self._is_test_function(node, decorators),
                docstring=ast.get_docstring(node),
                signature=self._signature(node),
                returns=_unparse(node.returns),
                decorators=decorators,
                parent_class=enclosing_class.qualified_name if enclosing_class else None,
                complexity=_cyclomatic_complexity(node),
            )
        )

        # Decorators and default values are evaluated in the *enclosing* scope.
        for decorator in node.decorator_list:
            self._visit(decorator)
        for default in [*node.args.defaults, *[d for d in node.args.kw_defaults if d]]:
            self._visit(default)

        fn_scope = self.scope.add_child(
            Scope(
                kind="function",
                qualified_name=qn,
                scope_id=self._new_scope_id(),
                qn_prefix=f"{qn}.<locals>",
                module_name=self.file.module_name,
                file_path=self.file.path,
            )
        )
        self._bind_parameters(fn_scope, node.args)

        self._scope_stack.append(fn_scope)
        self._caller_stack.append(qn)
        self._visit_body(node.body)
        self._caller_stack.pop()
        self._scope_stack.pop()

    def _handle_class(self, node: ast.ClassDef) -> None:
        qn = f"{self.scope.qn_prefix}.{node.name}" if self.scope.qn_prefix else node.name
        bases = [b for b in (_unparse(b) for b in node.bases) if b]
        decorators = [d for d in (_unparse(d) for d in node.decorator_list) if d]

        self.scope.bind(DefBinding(name=node.name, qualified_name=qn, kind="class"))

        self.symbols.append(
            SymbolDef(
                qualified_name=qn,
                name=node.name,
                kind=SymbolKind.CLASS,
                file_path=self.file.path,
                module_name=self.file.module_name,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                is_public=_is_public(node.name),
                docstring=ast.get_docstring(node),
                decorators=decorators,
                bases=bases,
                complexity=1,
            )
        )

        for extra in [*node.decorator_list, *node.bases, *node.keywords]:
            self._visit(extra)

        cls_scope = self.scope.add_child(
            Scope(
                kind="class",
                qualified_name=qn,
                scope_id=self._new_scope_id(),
                qn_prefix=qn,
                class_bases=bases,
                module_name=self.file.module_name,
                file_path=self.file.path,
            )
        )

        self._scope_stack.append(cls_scope)
        self._visit_body(node.body)
        self._scope_stack.pop()

    def _signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = _unparse(node.args) or ""
        returns = _unparse(node.returns)
        return f"{node.name}({args})" + (f" -> {returns}" if returns else "")

    def _is_test_function(self, node: ast.AST, decorators: list[str]) -> bool:
        name = getattr(node, "name", "")
        if name.startswith("test_"):
            return True
        if self.is_test_file and name.startswith("test"):
            return True
        return any("pytest" in d for d in decorators)

    def _bind_parameters(self, scope: Scope, args: ast.arguments) -> None:
        all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)
        for arg in all_args:
            scope.bind(ParamBinding(name=arg.arg, annotation=_unparse(arg.annotation)))

    # -- assignments -------------------------------------------------------------------

    def _handle_assign(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        annotation = _unparse(node.annotation) if isinstance(node, ast.AnnAssign) else None

        constructor: str | None = None
        if node.value is not None and isinstance(node.value, ast.Call):
            parts, ok = flatten_callee(node.value.func)
            if ok and parts:
                constructor = ".".join(parts)

        # A literal tells us the type outright: `data = {}` then `data.get(...)` is a call
        # on `dict`, which is a stdlib call rather than an unknown.
        if annotation is None and node.value is not None:
            annotation = _literal_type_name(node.value)

        for target in targets:
            if self._record_self_attribute(target, annotation, constructor):
                continue
            if not isinstance(target, ast.Name):
                self._bind_target_opaque(target)
                continue
            if annotation:
                # An explicit annotation is the programmer stating the type; trust it over
                # whatever the right-hand side looks like.
                self.scope.bind(ParamBinding(name=target.id, annotation=annotation))
            elif constructor:
                self.scope.bind(AssignBinding(name=target.id, constructor=constructor))
            else:
                self.scope.bind(OpaqueBinding(name=target.id))

    def _record_self_attribute(
        self, target: ast.expr, annotation: str | None, constructor: str | None
    ) -> bool:
        """Record `self.x = Foo()` / `self.x: Foo` on the enclosing class scope."""
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in ("self", "cls")
        ):
            return False

        cls_scope = self.scope.enclosing_class()
        if cls_scope is None:
            return False

        if annotation:
            cls_scope.instance_attributes[target.attr] = ParamBinding(
                name=target.attr, annotation=annotation
            )
        elif constructor:
            cls_scope.instance_attributes[target.attr] = AssignBinding(
                name=target.attr, constructor=constructor
            )
        else:
            cls_scope.instance_attributes.setdefault(
                target.attr, OpaqueBinding(name=target.attr)
            )
        return True

    def _bind_target_opaque(self, target: ast.expr) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                self.scope.bind(OpaqueBinding(name=node.id))

    # -- calls -------------------------------------------------------------------------

    def _handle_call(self, node: ast.Call) -> None:
        if _is_super_call(node.func):
            assert isinstance(node.func, ast.Attribute)
            self.raw_calls.append(
                RawCall(
                    caller_qn=self.caller_qn,
                    scope_id=self.scope.scope_id,
                    file_path=self.file.path,
                    line=node.lineno,
                    dotted_parts=[node.func.attr],
                    is_super_call=True,
                )
            )
            return

        parts, ok = flatten_callee(node.func)
        self.raw_calls.append(
            RawCall(
                caller_qn=self.caller_qn,
                scope_id=self.scope.scope_id,
                file_path=self.file.path,
                line=node.lineno,
                dotted_parts=parts if ok else [],
                is_super_call=False,
            )
        )
