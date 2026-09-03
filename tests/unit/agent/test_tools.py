"""AgentTools with a fake Neo4jAdapter — pure logic (result shaping, difflib
suggestions, direct-child filtering), no live Neo4j/Qdrant required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from codeatlas.agent.tools import AgentTools


class FakeAdapter:
    """Returns canned rows from a queue, one list per `execute_read` call, in order."""

    def __init__(self, responses: list[list[dict]]):
        self._responses = list(responses)
        self.queries: list[tuple[str, dict]] = []

    def execute_read(self, query: str, params: dict | None = None) -> list[dict]:
        self.queries.append((query, params or {}))
        if not self._responses:
            raise AssertionError(f"FakeAdapter ran out of canned responses at query: {query[:60]}")
        return self._responses.pop(0)


def make_tools(responses: list[list[dict]]) -> AgentTools:
    return AgentTools("repo1", adapter=FakeAdapter(responses))


# -- search_symbol -------------------------------------------------------------------


def test_search_symbol_found():
    tools = make_tools(
        [
            [
                {
                    "qualified_name": "pkg.mod.Foo",
                    "labels": ["Class"],
                    "signature": None,
                    "docstring": "A class.",
                    "file_path": "pkg/mod.py",
                    "start_line": 1,
                    "end_line": 10,
                }
            ]
        ]
    )
    result = tools.search_symbol("Foo")
    assert result["results"][0]["qualified_name"] == "pkg.mod.Foo"
    assert result["results"][0]["kind"] == "class"
    assert result["results"][0]["source"] == {"file_path": "pkg/mod.py", "start": 1, "end": 10}


def test_search_symbol_not_found_suggests_closest():
    # First call: exact match query -> empty. Second call (inside the suggestion
    # helper): list of all qualified_names to diff against.
    tools = make_tools([[], [{"qn": "pkg.mod.authenticate"}, {"qn": "pkg.mod.other"}]])
    result = tools.search_symbol("authX")
    assert "error" in result
    assert "authenticate" in result["error"]
    assert result["suggestions"][0] == "pkg.mod.authenticate"


def test_search_symbol_not_found_in_empty_repo():
    tools = make_tools([[], []])
    result = tools.search_symbol("anything")
    assert "error" in result
    assert "no indexed functions" in result["error"]


# -- get_callers / get_callees --------------------------------------------------------


def test_get_callers_shape():
    tools = make_tools(
        [
            [{"f": 1}],  # _symbol_exists
            [
                {
                    "qualified_name": "pkg.mod.caller",
                    "signature": "caller()",
                    "file_path": "pkg/mod.py",
                    "start_line": 5,
                    "end_line": 8,
                    "distance": 1,
                }
            ],
        ]
    )
    result = tools.get_callers("pkg.mod.target", depth=1)
    assert result["callers"][0]["qualified_name"] == "pkg.mod.caller"
    assert result["callers"][0]["source"] == {"file_path": "pkg/mod.py", "start": 5, "end": 8}


def test_get_callers_unknown_symbol_suggests_closest():
    tools = make_tools([[], [{"qn": "pkg.mod.real_func"}]])
    result = tools.get_callers("pkg.mod.typo_func")
    assert "error" in result
    assert "suggestions" in result


def test_get_callees_uses_callees_key():
    tools = make_tools(
        [
            [{"f": 1}],
            [
                {
                    "qualified_name": "pkg.mod.callee",
                    "signature": "callee()",
                    "file_path": "pkg/mod.py",
                    "start_line": 1,
                    "end_line": 2,
                    "distance": 1,
                }
            ],
        ]
    )
    result = tools.get_callees("pkg.mod.source_fn")
    assert "callees" in result
    assert "callers" not in result


def test_depth_is_clamped_to_range():
    """Doesn't crash on an absurd depth; just clamps it before building Cypher."""
    tools = make_tools([[{"f": 1}], []])
    tools.get_callers("pkg.mod.x", depth=999)
    # second query is the neighbor lookup; its query text should mention *1..5, not *1..999
    query_text = tools.adapter.queries[1][0]
    assert "*1..5" in query_text


# -- impact_analysis --------------------------------------------------------------------


def test_impact_analysis_shape():
    tools = make_tools(
        [
            [{"f": 1}],  # _symbol_exists
            [
                {
                    "qualified_name": "pkg.mod.impacted",
                    "file_path": "pkg/mod.py",
                    "start_line": 1,
                    "end_line": 5,
                    "distance": 1,
                }
            ],
            [{"qualified_name": "tests.test_mod.test_it", "file_path": "tests/test_mod.py"}],
        ]
    )
    result = tools.impact_analysis("pkg.mod.target")
    assert result["impacted_symbols"][0]["qualified_name"] == "pkg.mod.impacted"
    assert result["impacted_symbols"][0]["source"]["file_path"] == "pkg/mod.py"
    assert result["affected_tests"] == ["tests.test_mod.test_it"]


def test_impact_analysis_unknown_symbol():
    tools = make_tools([[], [{"qn": "pkg.mod.real"}]])
    result = tools.impact_analysis("pkg.mod.fake")
    assert "error" in result


# -- list_module_structure ---------------------------------------------------------------


def test_list_module_structure_filters_nested_and_splits_by_kind():
    tools = make_tools(
        [
            [
                {
                    "qualified_name": "pkg.mod.TopClass",
                    "labels": ["Class"],
                    "file_path": "pkg/mod.py",
                    "start_line": 1,
                    "end_line": 20,
                    "docstring": None,
                },
                {
                    "qualified_name": "pkg.mod.TopClass.method",  # nested, must be excluded
                    "labels": ["Function"],
                    "file_path": "pkg/mod.py",
                    "start_line": 2,
                    "end_line": 3,
                    "docstring": None,
                },
                {
                    "qualified_name": "pkg.mod.top_func",
                    "labels": ["Function"],
                    "file_path": "pkg/mod.py",
                    "start_line": 25,
                    "end_line": 30,
                    "docstring": "does a thing",
                },
            ]
        ]
    )
    result = tools.list_module_structure("pkg.mod")
    assert [c["qualified_name"] for c in result["classes"]] == ["pkg.mod.TopClass"]
    assert [f["qualified_name"] for f in result["functions"]] == ["pkg.mod.top_func"]


def test_list_module_structure_empty_module():
    tools = make_tools([[]])
    result = tools.list_module_structure("pkg.nonexistent")
    assert "error" in result


# -- read_source ------------------------------------------------------------------------


def _fake_qdrant_point(payload: dict):
    p = MagicMock()
    p.payload = payload
    return p


def test_read_source_concatenates_overlapping_chunks():
    tools = make_tools([])
    points = [
        _fake_qdrant_point(
            {"content": "def a():\n    pass", "start_line": 1, "end_line": 2, "part": 0}
        ),
        _fake_qdrant_point(
            {"content": "def b():\n    pass", "start_line": 3, "end_line": 4, "part": 0}
        ),
    ]
    with patch("codeatlas.infrastructure.db.qdrant.connection.scroll", return_value=(points, None)):
        result = tools.read_source("pkg/mod.py", 1, 4)

    assert "def a()" in result["content"]
    assert "def b()" in result["content"]
    assert result["source"] == {"file_path": "pkg/mod.py", "start": 1, "end": 4}


def test_read_source_no_covering_chunk_is_a_real_miss():
    tools = make_tools([])
    with patch("codeatlas.infrastructure.db.qdrant.connection.scroll", return_value=([], None)):
        result = tools.read_source("pkg/mod.py", 100, 200)
    assert "error" in result
    assert "No indexed chunk" in result["error"]
