from __future__ import annotations

from codeatlas.agent.harness import MAX_TOOL_CALLS, ToolCallHarness


def test_budget_exceeded_after_max_calls():
    h = ToolCallHarness(max_calls=3)
    assert not h.budget_exceeded
    h.record("search_symbol", {"name": "a"})
    h.record("search_symbol", {"name": "b"})
    assert not h.budget_exceeded
    h.record("search_symbol", {"name": "c"})
    assert h.budget_exceeded


def test_default_ceiling_matches_spec():
    assert MAX_TOOL_CALLS == 8


def test_first_call_has_no_warning():
    h = ToolCallHarness()
    assert h.record("search_symbol", {"name": "foo"}) is None


def test_identical_second_call_warns():
    h = ToolCallHarness()
    h.record("search_symbol", {"name": "foo"})
    warning = h.record("search_symbol", {"name": "foo"})
    assert warning is not None
    assert "search_symbol" in warning


def test_different_args_do_not_trigger_loop_detection():
    h = ToolCallHarness()
    h.record("search_symbol", {"name": "foo"})
    assert h.record("search_symbol", {"name": "bar"}) is None


def test_different_tool_same_args_do_not_trigger_loop_detection():
    h = ToolCallHarness()
    h.record("search_symbol", {"name": "foo"})
    assert h.record("get_callers", {"name": "foo"}) is None


def test_arg_order_does_not_matter_for_loop_detection():
    h = ToolCallHarness()
    h.record("get_callers", {"qualified_name": "a.b", "depth": 2})
    warning = h.record("get_callers", {"depth": 2, "qualified_name": "a.b"})
    assert warning is not None


def test_calls_used_counts_every_call_including_repeats():
    h = ToolCallHarness()
    h.record("search_symbol", {"name": "foo"})
    h.record("search_symbol", {"name": "foo"})
    assert h.calls_used == 2
