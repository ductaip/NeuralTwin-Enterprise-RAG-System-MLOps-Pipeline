"""AtlasState — CODEATLAS_SPEC.md §2.3, verbatim.

Both orchestrators (custom_react.py, langgraph_qa.py) read and write this same shape so
Bảng B (spec §3.3) compares orchestration, not state layout.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict


class AtlasState(TypedDict):
    # Input
    query: str
    mode: Literal["qa", "refactor"]
    repo_id: str
    # Working
    plan: str
    evidence: Annotated[list[dict], add]  # reducer: gộp thay vì ghi đè
    tool_budget_used: int
    # Refactor branch (Phase 4)
    impacted_symbols: list[str]
    affected_tests: list[str]
    patch: str | None
    test_output: str | None
    repair_iteration: int
    # Output
    answer: str
    citations: list[dict]


def initial_state(query: str, repo_id: str, mode: Literal["qa", "refactor"] = "qa") -> AtlasState:
    return AtlasState(
        query=query,
        mode=mode,
        repo_id=repo_id,
        plan="",
        evidence=[],
        tool_budget_used=0,
        impacted_symbols=[],
        affected_tests=[],
        patch=None,
        test_output=None,
        repair_iteration=0,
        answer="",
        citations=[],
    )
