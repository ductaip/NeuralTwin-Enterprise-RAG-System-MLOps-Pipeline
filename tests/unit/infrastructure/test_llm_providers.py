"""Pure-logic tests for the LLM providers: no network calls.

Live behaviour (real Groq call, real header parsing, real cache hit) is verified
separately against the actual API — see `docs/PHASE2_RESULTS.md`.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from codeatlas.infrastructure.llm.cache import cache_key
from codeatlas.infrastructure.llm.groq_provider import GroqProvider, _parse_duration_seconds


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2s", 2.0),
        ("907ms", 0.907),
        ("1m30s", 90.0),
        ("1m30.5s", 90.5),
        ("1h", 3600.0),
        ("0.244s", 0.244),
        (None, None),
        ("", None),
        ("garbage", None),
    ],
)
def test_parse_duration_seconds(raw, expected):
    result = _parse_duration_seconds(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_cache_key_is_stable_and_sensitive_to_every_field():
    base = cache_key("groq", "llama-3.3-70b-versatile", "hello", 0.0, 0)
    assert base == cache_key("groq", "llama-3.3-70b-versatile", "hello", 0.0, 0)
    assert base != cache_key("groq", "llama-3.3-70b-versatile", "different prompt", 0.0, 0)
    assert base != cache_key("groq", "other-model", "hello", 0.0, 0)
    assert base != cache_key("groq", "llama-3.3-70b-versatile", "hello", 0.7, 0)
    assert base != cache_key("modal_vllm", "llama-3.3-70b-versatile", "hello", 0.0, 0)


def test_missing_api_key_raises_immediately(monkeypatch):
    from codeatlas.domain.exceptions import LLMGenerationError
    from codeatlas.settings import settings

    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    with pytest.raises(LLMGenerationError):
        GroqProvider(api_key=None)


def test_to_openai_messages_handles_langchain_and_dict_and_str():
    formatted = GroqProvider._to_openai_messages("plain string")
    assert formatted == [{"role": "user", "content": "plain string"}]

    formatted = GroqProvider._to_openai_messages(
        [SystemMessage(content="sys"), HumanMessage(content="hi")]
    )
    assert formatted == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]

    formatted = GroqProvider._to_openai_messages([{"role": "user", "content": "raw dict"}])
    assert formatted == [{"role": "user", "content": "raw dict"}]
