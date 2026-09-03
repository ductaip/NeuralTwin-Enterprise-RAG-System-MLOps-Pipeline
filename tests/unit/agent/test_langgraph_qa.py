from __future__ import annotations

import pytest

from codeatlas.agent.langgraph_qa import _extract_symbol_mention


@pytest.mark.parametrize(
    "query,expected",
    [
        ("who calls APIRouter.add_api_route", "APIRouter.add_api_route"),
        ("what does BackgroundTasks.add_task do", "BackgroundTasks.add_task"),
        ("callers of fastapi.encoders.jsonable_encoder", "fastapi.encoders.jsonable_encoder"),
        ("who calls jsonable_encoder", "jsonable_encoder"),
        ("impact of changing get_current_user", "get_current_user"),
    ],
)
def test_extracts_the_symbol_not_the_whole_sentence(query, expected):
    """The bug this guards against: passing the raw sentence to `search_symbol` (an
    exact-match lookup) never matches anything — live-verified miss on "who calls
    APIRouter.add_api_route", which fell through to a nonsense difflib guess."""
    assert _extract_symbol_mention(query) == expected


def test_prefers_dotted_identifier_over_bare_word():
    mention = _extract_symbol_mention("who calls x.y.z near some_other_word")
    assert mention == "x.y.z"


def test_no_identifier_returns_none():
    assert _extract_symbol_mention("who calls it") is None
