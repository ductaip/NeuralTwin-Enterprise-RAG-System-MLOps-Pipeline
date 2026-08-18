from __future__ import annotations

from typing import get_type_hints

from codeatlas.agent.state import AtlasState, initial_state


def test_initial_state_has_every_spec_field():
    state = initial_state("q", "repo1")
    expected_keys = {
        "query", "mode", "repo_id", "plan", "evidence", "tool_budget_used",
        "impacted_symbols", "affected_tests", "patch", "test_output",
        "repair_iteration", "answer", "citations",
    }
    assert set(state.keys()) == expected_keys


def test_initial_state_defaults():
    state = initial_state("how does X work", "fastapi")
    assert state["query"] == "how does X work"
    assert state["repo_id"] == "fastapi"
    assert state["mode"] == "qa"
    assert state["evidence"] == []
    assert state["tool_budget_used"] == 0
    assert state["patch"] is None
    assert state["repair_iteration"] == 0


def test_initial_state_mode_override():
    assert initial_state("q", "r", mode="refactor")["mode"] == "refactor"


def test_evidence_reducer_is_additive_not_overwrite():
    """This is the exact property spec §2.3 calls out: `Annotated[list[dict], add]`
    so a second node's evidence appends rather than clobbers the first's."""
    hints = get_type_hints(AtlasState, include_extras=True)
    evidence_type = hints["evidence"]
    reducer = evidence_type.__metadata__[0]

    merged = reducer([{"a": 1}], [{"b": 2}])
    assert merged == [{"a": 1}, {"b": 2}]
