"""Hard cases for `SymbolResolver`.

Every case in `docs/codeatlas_roadmap.md` Phase 1 item 3 gets at least one test, plus the
two rules that must never regress: external calls are a *success* bucket, and a name that
is locally rebound must not resolve to the module it shadows.
"""

from __future__ import annotations

import pytest

from codeatlas.ingestion.models import ResolutionStatus, SourceFile
from codeatlas.ingestion.python_parser import PythonParser
from codeatlas.ingestion.symbol_resolver import SymbolResolver


def _module_name(path: str) -> str:
    parts = path.split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def build(files: dict[str, str]) -> tuple[SymbolResolver, "object"]:
    parser = PythonParser()
    modules = []
    for path, content in files.items():
        source = SourceFile(
            path=path,
            module_name=_module_name(path),
            content=content,
            sha="deadbeef",
            loc=content.count("\n") + 1,
        )
        modules.append(parser.parse(source))
    resolver = SymbolResolver(modules)
    return resolver, resolver.resolve_all()


def edges_from(result, caller: str) -> list:
    return [e for e in result.call_edges if e.caller_qn == caller]


def targets_from(result, caller: str) -> set[str]:
    return {e.callee_qn for e in edges_from(result, caller)}


def reason_for(result, caller: str, callee: str) -> str:
    for edge in result.call_edges:
        if edge.caller_qn == caller and edge.callee_qn == callee:
            return edge.reason
    raise AssertionError(f"no edge {caller} -> {callee}; have {targets_from(result, caller)}")


def unresolved_reasons(result) -> list[str]:
    return [s["reason"] for s in result.report.unresolved_samples]


# --------------------------------------------------------------------------------------
# 1. Bare names and local definitions
# --------------------------------------------------------------------------------------


def test_bare_call_in_same_module():
    _, result = build({"pkg/mod.py": "def helper():\n    pass\n\ndef caller():\n    helper()\n"})
    assert "pkg.mod.helper" in targets_from(result, "pkg.mod.caller")


def test_forward_reference_resolves():
    """A call to a function defined later in the file must still resolve."""
    _, result = build({"pkg/mod.py": "def caller():\n    later()\n\ndef later():\n    pass\n"})
    assert "pkg.mod.later" in targets_from(result, "pkg.mod.caller")


# --------------------------------------------------------------------------------------
# 2. Imports
# --------------------------------------------------------------------------------------


def test_import_dotted_with_alias():
    """`import x.y as z` then `z.foo()`."""
    _, result = build(
        {
            "x/__init__.py": "",
            "x/y.py": "def foo():\n    pass\n",
            "app.py": "import x.y as z\n\ndef caller():\n    z.foo()\n",
        }
    )
    assert "x.y.foo" in targets_from(result, "app.caller")


def test_import_dotted_without_alias_binds_top_name():
    """`import x.y` binds `x`, and `x.y.foo()` must walk through to the real module."""
    _, result = build(
        {
            "x/__init__.py": "",
            "x/y.py": "def foo():\n    pass\n",
            "app.py": "import x.y\n\ndef caller():\n    x.y.foo()\n",
        }
    )
    assert "x.y.foo" in targets_from(result, "app.caller")


def test_from_import_symbol():
    _, result = build(
        {
            "a/__init__.py": "",
            "a/b.py": "def c():\n    pass\n",
            "app.py": "from a.b import c\n\ndef caller():\n    c()\n",
        }
    )
    assert "a.b.c" in targets_from(result, "app.caller")


def test_from_import_with_alias():
    _, result = build(
        {
            "a/__init__.py": "",
            "a/b.py": "def c():\n    pass\n",
            "app.py": "from a.b import c as renamed\n\ndef caller():\n    renamed()\n",
        }
    )
    assert "a.b.c" in targets_from(result, "app.caller")


def test_relative_import_single_dot():
    _, result = build(
        {
            "pkg/__init__.py": "",
            "pkg/helpers.py": "def util():\n    pass\n",
            "pkg/main.py": "from . import helpers\n\ndef caller():\n    helpers.util()\n",
        }
    )
    assert "pkg.helpers.util" in targets_from(result, "pkg.main.caller")


def test_relative_import_symbol_from_sibling():
    _, result = build(
        {
            "pkg/__init__.py": "",
            "pkg/helpers.py": "def util():\n    pass\n",
            "pkg/main.py": "from .helpers import util\n\ndef caller():\n    util()\n",
        }
    )
    assert "pkg.helpers.util" in targets_from(result, "pkg.main.caller")


def test_relative_import_two_dots():
    _, result = build(
        {
            "pkg/__init__.py": "",
            "pkg/tools.py": "def shared():\n    pass\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/deep.py": "from ..tools import shared\n\ndef caller():\n    shared()\n",
        }
    )
    assert "pkg.tools.shared" in targets_from(result, "pkg.sub.deep.caller")


def test_reexport_through_package_init():
    """`from pkg import thing` where `pkg/__init__.py` re-exports it from a submodule."""
    _, result = build(
        {
            "pkg/__init__.py": "from .impl import thing\n",
            "pkg/impl.py": "def thing():\n    pass\n",
            "app.py": "from pkg import thing\n\ndef caller():\n    thing()\n",
        }
    )
    assert "pkg.impl.thing" in targets_from(result, "app.caller")
    assert reason_for(result, "app.caller", "pkg.impl.thing") == "via_reexport"


# --------------------------------------------------------------------------------------
# 3. self / super / inheritance
# --------------------------------------------------------------------------------------


def test_self_method_own_class():
    _, result = build(
        {
            "pkg/mod.py": (
                "class Service:\n"
                "    def run(self):\n"
                "        self.helper()\n"
                "    def helper(self):\n"
                "        pass\n"
            )
        }
    )
    assert "pkg.mod.Service.helper" in targets_from(result, "pkg.mod.Service.run")


def test_self_method_inherited():
    _, result = build(
        {
            "pkg/base.py": "class Base:\n    def shared(self):\n        pass\n",
            "pkg/__init__.py": "",
            "pkg/child.py": (
                "from .base import Base\n\n"
                "class Child(Base):\n"
                "    def run(self):\n"
                "        self.shared()\n"
            ),
        }
    )
    assert "pkg.base.Base.shared" in targets_from(result, "pkg.child.Child.run")
    assert reason_for(result, "pkg.child.Child.run", "pkg.base.Base.shared") == "self_attr_inherited"


def test_super_call_resolves_to_base():
    _, result = build(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": "class Base:\n    def setup(self):\n        pass\n",
            "pkg/child.py": (
                "from .base import Base\n\n"
                "class Child(Base):\n"
                "    def setup(self):\n"
                "        super().setup()\n"
            ),
        }
    )
    assert "pkg.base.Base.setup" in targets_from(result, "pkg.child.Child.setup")


def test_self_attr_on_external_base_is_external_not_unresolved():
    """Inheriting from something outside the index is a known unknown, not a failure."""
    _, result = build(
        {
            "app.py": (
                "from pydantic import BaseModel\n\n"
                "class Config(BaseModel):\n"
                "    def go(self):\n"
                "        self.model_dump()\n"
            )
        }
    )
    assert result.report.external >= 1
    assert "self_attr_not_found" not in unresolved_reasons(result)


# --------------------------------------------------------------------------------------
# 4. Nesting
# --------------------------------------------------------------------------------------


def test_nested_function_uses_locals_marker():
    resolver, _ = build(
        {"pkg/mod.py": "def outer():\n    def inner():\n        pass\n    inner()\n"}
    )
    assert "pkg.mod.outer.<locals>.inner" in resolver.symbol_table


def test_two_nested_functions_with_same_name_do_not_collide():
    """The failure this guards against is silent: two `inner`s merging into one node."""
    resolver, _ = build(
        {
            "pkg/mod.py": (
                "def first():\n"
                "    def inner():\n"
                "        pass\n"
                "    inner()\n"
                "\n"
                "def second():\n"
                "    def inner():\n"
                "        pass\n"
                "    inner()\n"
            )
        }
    )
    assert "pkg.mod.first.<locals>.inner" in resolver.symbol_table
    assert "pkg.mod.second.<locals>.inner" in resolver.symbol_table


def test_nested_call_targets_correct_inner():
    _, result = build(
        {
            "pkg/mod.py": (
                "def first():\n"
                "    def inner():\n"
                "        pass\n"
                "    inner()\n"
                "\n"
                "def second():\n"
                "    def inner():\n"
                "        pass\n"
                "    inner()\n"
            )
        }
    )
    assert targets_from(result, "pkg.mod.first") == {"pkg.mod.first.<locals>.inner"}
    assert targets_from(result, "pkg.mod.second") == {"pkg.mod.second.<locals>.inner"}


def test_nested_class_method():
    resolver, _ = build(
        {"pkg/mod.py": "def factory():\n    class Inner:\n        def go(self):\n            pass\n"}
    )
    assert "pkg.mod.factory.<locals>.Inner.go" in resolver.symbol_table


# --------------------------------------------------------------------------------------
# 5. Instance typing (best effort)
# --------------------------------------------------------------------------------------


def test_method_via_local_instance_variable():
    _, result = build(
        {
            "pkg/mod.py": (
                "class Engine:\n"
                "    def start(self):\n"
                "        pass\n"
                "\n"
                "def caller():\n"
                "    engine = Engine()\n"
                "    engine.start()\n"
            )
        }
    )
    assert "pkg.mod.Engine.start" in targets_from(result, "pkg.mod.caller")
    edge = next(
        e for e in result.call_edges
        if e.caller_qn == "pkg.mod.caller" and e.callee_qn == "pkg.mod.Engine.start"
    )
    assert edge.confidence == pytest.approx(0.5), "inferred instance typing is best-effort"


def test_method_via_annotated_parameter_is_higher_confidence():
    _, result = build(
        {
            "pkg/mod.py": (
                "class Engine:\n"
                "    def start(self):\n"
                "        pass\n"
                "\n"
                "def caller(engine: Engine):\n"
                "    engine.start()\n"
            )
        }
    )
    edge = next(
        e for e in result.call_edges
        if e.caller_qn == "pkg.mod.caller" and e.callee_qn == "pkg.mod.Engine.start"
    )
    assert edge.confidence == pytest.approx(0.9), "an annotation is the author stating the type"


def test_reassigned_variable_falls_back_to_opaque():
    """`x = Foo()` then `x = something_else()` must not keep claiming `x` is a `Foo`."""
    _, result = build(
        {
            "pkg/mod.py": (
                "class Engine:\n"
                "    def start(self):\n"
                "        pass\n"
                "\n"
                "def unknown():\n"
                "    pass\n"
                "\n"
                "def caller():\n"
                "    engine = Engine()\n"
                "    engine = unknown()\n"
                "    engine.start()\n"
            )
        }
    )
    assert "pkg.mod.Engine.start" not in targets_from(result, "pkg.mod.caller")


# --------------------------------------------------------------------------------------
# 6. Shadowing
# --------------------------------------------------------------------------------------


def test_local_rebinding_shadows_module_import():
    """`import json` at module level, `json = load()` locally: the call is NOT stdlib json."""
    _, result = build(
        {
            "app.py": (
                "import json\n\n"
                "def load():\n"
                "    pass\n"
                "\n"
                "def caller():\n"
                "    json = load()\n"
                "    json.dumps({})\n"
            )
        }
    )
    # The invariant that matters: the call did not silently become stdlib `json.dumps`.
    assert result.call_edges == [
        e for e in result.call_edges if e.callee_qn == "app.load"
    ], "the locally rebound name must not resolve through the shadowed import"
    assert "assign_from_function_return" in unresolved_reasons(result)


def test_parameter_shadows_module_level_function():
    _, result = build(
        {
            "app.py": (
                "def handler():\n"
                "    pass\n"
                "\n"
                "def caller(handler):\n"
                "    handler()\n"
            )
        }
    )
    assert "app.handler" not in targets_from(result, "app.caller")


def test_class_body_name_not_visible_from_method():
    """Python's LEGB skips the class body for nested functions; the resolver must too."""
    _, result = build(
        {
            "pkg/mod.py": (
                "def helper():\n"
                "    pass\n"
                "\n"
                "class Thing:\n"
                "    def helper(self):\n"
                "        pass\n"
                "    def run(self):\n"
                "        helper()\n"
            )
        }
    )
    assert "pkg.mod.helper" in targets_from(result, "pkg.mod.Thing.run")
    assert "pkg.mod.Thing.helper" not in targets_from(result, "pkg.mod.Thing.run")


# --------------------------------------------------------------------------------------
# 7. Buckets: external vs unresolved
# --------------------------------------------------------------------------------------


def test_stdlib_call_is_external_not_unresolved():
    _, result = build({"app.py": "import os\n\ndef caller():\n    os.path.join('a', 'b')\n"})
    assert result.report.external >= 1
    assert result.report.unresolved == 0


def test_builtin_call_is_external():
    _, result = build({"app.py": "def caller():\n    len([1, 2])\n"})
    assert result.report.external >= 1
    assert result.report.unresolved == 0


def test_external_calls_excluded_from_resolve_rate_denominator():
    _, result = build(
        {
            "app.py": (
                "import os\n\n"
                "def helper():\n"
                "    pass\n"
                "\n"
                "def caller():\n"
                "    os.getcwd()\n"
                "    helper()\n"
            )
        }
    )
    # One internal call, one external, nothing unresolved -> a clean 100% internal rate.
    assert result.report.internal_call_sites == result.report.internal + result.report.unresolved
    assert result.report.internal_resolve_rate == pytest.approx(1.0)


def test_dynamic_callee_is_counted_not_hidden():
    """`handlers[key]()` cannot be named — it stays in the denominator."""
    _, result = build({"app.py": "def caller(handlers, key):\n    handlers[key]()\n"})
    assert "dynamic_callee" in unresolved_reasons(result)
    assert result.report.unresolved == 1


def test_unknown_name_is_unresolved_not_guessed():
    _, result = build({"app.py": "def caller():\n    mystery_function()\n"})
    assert "name_not_found" in unresolved_reasons(result)
    assert result.call_edges == []


# --------------------------------------------------------------------------------------
# 8. Instantiation and derived edges
# --------------------------------------------------------------------------------------


def test_instantiation_points_at_init():
    _, result = build(
        {
            "pkg/mod.py": (
                "class Engine:\n"
                "    def __init__(self):\n"
                "        pass\n"
                "\n"
                "def caller():\n"
                "    Engine()\n"
            )
        }
    )
    assert "pkg.mod.Engine.__init__" in targets_from(result, "pkg.mod.caller")


def test_instantiation_without_init_still_emits_edge():
    """Losing this edge would hide every constructor call from impact analysis."""
    _, result = build(
        {"pkg/mod.py": "class Plain:\n    pass\n\ndef caller():\n    Plain()\n"}
    )
    assert "pkg.mod.Plain" in targets_from(result, "pkg.mod.caller")


def test_test_function_produces_test_edge():
    _, result = build(
        {
            "pkg/mod.py": "def target():\n    pass\n",
            "tests/test_mod.py": "from pkg.mod import target\n\ndef test_target():\n    target()\n",
        }
    )
    assert any(
        e.test_qn == "tests.test_mod.test_target" and e.target_qn == "pkg.mod.target"
        for e in result.test_edges
    )


def test_value_derived_from_external_object_is_external():
    """`client = TestClient()` -> `response = client.get()` -> `response.json()`.

    Every link is outside the index, so the last call is external rather than a failure.
    """
    _, result = build(
        {
            "app.py": (
                "from starlette.testclient import TestClient\n\n"
                "def test_read():\n"
                "    client = TestClient(None)\n"
                "    response = client.get('/')\n"
                "    response.json()\n"
            )
        }
    )
    assert "assign_from_method_return" not in unresolved_reasons(result)
    assert result.report.external >= 2


def test_literal_assignment_is_typed_as_builtin():
    """`data = {}` then `data.get(...)` is a stdlib call, not an unknown."""
    _, result = build({"app.py": "def caller():\n    data = {}\n    data.get('k')\n"})
    assert "opaque_binding" not in unresolved_reasons(result)
    assert result.report.external >= 1


def test_return_annotation_types_the_result():
    _, result = build(
        {
            "pkg/mod.py": (
                "class Engine:\n"
                "    def start(self):\n"
                "        pass\n"
                "\n"
                "def build() -> Engine:\n"
                "    return Engine()\n"
                "\n"
                "def caller():\n"
                "    engine = build()\n"
                "    engine.start()\n"
            )
        }
    )
    assert "pkg.mod.Engine.start" in targets_from(result, "pkg.mod.caller")
    assert reason_for(result, "pkg.mod.caller", "pkg.mod.Engine.start") == "inferred_return_own"


def test_pytest_fixture_is_not_marked_as_a_test():
    """`pytest tests/x.py::get_client` is not a runnable node id — fixtures are not tests."""
    resolver, _ = build(
        {
            "tests/test_mod.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def get_client():\n"
                "    return 1\n"
                "\n"
                "def test_real():\n"
                "    pass\n"
            )
        }
    )
    assert resolver.symbol_table["tests.test_mod.get_client"].is_test is False
    assert resolver.symbol_table["tests.test_mod.test_real"].is_test is True


def test_inheritance_edges_split_internal_and_external():
    _, result = build(
        {
            "app.py": (
                "from pydantic import BaseModel\n\n"
                "class Local:\n"
                "    pass\n"
                "\n"
                "class Child(Local):\n"
                "    pass\n"
                "\n"
                "class Model(BaseModel):\n"
                "    pass\n"
            )
        }
    )
    internal = [e for e in result.inheritance_edges if e.status is ResolutionStatus.INTERNAL]
    external = [e for e in result.inheritance_edges if e.status is ResolutionStatus.EXTERNAL]
    assert any(e.child_qn == "app.Child" and e.parent_qn == "app.Local" for e in internal)
    assert any(e.child_qn == "app.Model" for e in external)
