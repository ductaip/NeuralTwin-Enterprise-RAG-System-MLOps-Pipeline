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


class _FakeResponse:
    def __init__(self, headers: dict):
        self.headers = headers


class _FakeAPIStatusError:
    """Duck-typed stand-in for `openai.APIStatusError` — only `.response.headers` is
    read by `_backoff_seconds`."""

    def __init__(self, headers: dict):
        self.response = _FakeResponse(headers)


def test_backoff_waits_for_whichever_bucket_refills_sooner(monkeypatch):
    """Live-verified: `remaining_requests` stays high (deep daily pool) while
    `remaining_tokens` depletes fast in a chatty loop — reset-tokens ~4-5s vs
    reset-requests ~20+ min on the very same response. Picking the wrong one means
    waiting 20 minutes for a limit that was never the actual problem."""
    from codeatlas.settings import settings

    monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-unit-test")
    provider = GroqProvider(use_cache=False)

    error = _FakeAPIStatusError(
        {"x-ratelimit-reset-requests": "20m9.6s", "x-ratelimit-reset-tokens": "4.3s"}
    )
    provider.last_rate_limit = provider._parse_rate_limit_headers(error.response.headers)

    wait = provider._backoff_seconds(error, attempt=0)
    assert wait == pytest.approx(4.3)


def test_backoff_falls_back_to_only_available_bucket(monkeypatch):
    from codeatlas.settings import settings

    monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-unit-test")
    provider = GroqProvider(use_cache=False)

    error = _FakeAPIStatusError({"x-ratelimit-reset-requests": "12s"})
    provider.last_rate_limit = provider._parse_rate_limit_headers(error.response.headers)

    assert provider._backoff_seconds(error, attempt=0) == pytest.approx(12.0)


def test_retry_after_header_wins_over_rate_limit_reset(monkeypatch):
    from codeatlas.settings import settings

    monkeypatch.setattr(settings, "GROQ_API_KEY", "fake-key-for-unit-test")
    provider = GroqProvider(use_cache=False)

    error = _FakeAPIStatusError(
        {"retry-after": "2", "x-ratelimit-reset-requests": "20m", "x-ratelimit-reset-tokens": "9s"}
    )
    provider.last_rate_limit = provider._parse_rate_limit_headers(error.response.headers)

    assert provider._backoff_seconds(error, attempt=0) == pytest.approx(2.0)


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
