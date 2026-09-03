"""Groq client: OpenAI-compatible endpoint, runtime rate-limit backoff.

Live-demo backend per spec §2.6 — 8 sequential tool calls per query is latency-bound,
and Groq runs 300-1000 tokens/sec.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from loguru import logger
from openai import APIStatusError, OpenAI

from codeatlas.domain.exceptions import LLMGenerationError
from codeatlas.infrastructure.llm.cache import cache_key, get_cache
from codeatlas.settings import settings

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def _parse_duration_seconds(value: str | None) -> float | None:
    """Parse a Go-style duration string ('2s', '1m30.5s', '907ms') into seconds.

    Groq mirrors OpenAI's `x-ratelimit-reset-*` header format. Returns None rather than
    guessing when the string doesn't match — an unparseable header should fall back to
    the exponential-backoff path, not silently wait zero seconds.
    """
    if not value:
        return None
    matches = _DURATION_RE.findall(value)
    if not matches:
        return None
    unit_seconds = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(amount) * unit_seconds[unit] for amount, unit in matches)


@dataclass
class RateLimitState:
    limit_requests: int | None = None
    limit_tokens: int | None = None
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests_seconds: float | None = None
    reset_tokens_seconds: float | None = None
    observed_at: float = 0.0


class GroqProvider:
    """Duck-typed like `VLLMClient`: `.invoke(messages) -> AIMessage` for LCEL chains,
    plus `.generate(prompt)` for direct use in HyDE/contextual retrieval.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 5,
        seed: int | None = 0,
        use_cache: bool = True,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            raise LLMGenerationError(
                "GROQ_API_KEY is not set. Add it to .env or pass api_key explicitly."
            )
        self.model = model or settings.GROQ_MODEL_ID
        self.max_retries = max_retries
        self.seed = seed
        self.use_cache = use_cache
        self._client = OpenAI(api_key=self.api_key, base_url=settings.GROQ_BASE_URL)
        self._cache = get_cache()
        self.last_rate_limit = RateLimitState()

    def _parse_rate_limit_headers(self, headers: Any) -> RateLimitState:
        def _int(name: str) -> int | None:
            value = headers.get(name)
            return int(value) if value is not None else None

        return RateLimitState(
            limit_requests=_int("x-ratelimit-limit-requests"),
            limit_tokens=_int("x-ratelimit-limit-tokens"),
            remaining_requests=_int("x-ratelimit-remaining-requests"),
            remaining_tokens=_int("x-ratelimit-remaining-tokens"),
            reset_requests_seconds=_parse_duration_seconds(headers.get("x-ratelimit-reset-requests")),
            reset_tokens_seconds=_parse_duration_seconds(headers.get("x-ratelimit-reset-tokens")),
            observed_at=time.monotonic(),
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        prompt_for_cache = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        key = cache_key("groq", self.model, prompt_for_cache, temperature, self.seed)
        if self.use_cache and key in self._cache:
            return self._cache[key]

        content = self._call_with_backoff(messages, max_tokens, temperature)

        if self.use_cache:
            self._cache[key] = content
        return content

    def _call_with_backoff(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                raw = self._client.chat.completions.with_raw_response.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=self.seed,
                )
                self.last_rate_limit = self._parse_rate_limit_headers(raw.headers)
                response = raw.parse()
                return response.choices[0].message.content or ""

            except APIStatusError as e:
                last_error = e
                if e.status_code != 429 and e.status_code < 500:
                    raise LLMGenerationError(f"Groq request failed ({e.status_code}): {e}") from e

                self.last_rate_limit = self._parse_rate_limit_headers(e.response.headers)
                wait = self._backoff_seconds(e, attempt)
                logger.warning(
                    f"Groq rate-limited/errored (attempt {attempt + 1}/{self.max_retries}), "
                    f"waiting {wait:.1f}s. remaining_requests="
                    f"{self.last_rate_limit.remaining_requests} "
                    f"remaining_tokens={self.last_rate_limit.remaining_tokens}"
                )
                time.sleep(wait)

        raise LLMGenerationError(
            f"Groq request failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def _backoff_seconds(self, error: APIStatusError, attempt: int) -> float:
        retry_after = error.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass

        # Wait for whichever bucket is closer to refilling, not requests unconditionally.
        # remaining_requests tends to stay high (deep daily pool) while remaining_tokens
        # depletes fast in a chatty loop — verified live: reset-tokens ~4-5s vs
        # reset-requests ~20+ min on the same response. Picking the wrong one means
        # waiting 20 minutes for a limit that was never actually the problem.
        candidates = [
            r
            for r in (self.last_rate_limit.reset_requests_seconds, self.last_rate_limit.reset_tokens_seconds)
            if r is not None
        ]
        if candidates:
            return min(candidates)

        base = min(2**attempt, 30)
        return base + random.uniform(0, base * 0.25)

    def invoke(self, messages: Any) -> AIMessage:
        formatted = self._to_openai_messages(messages)
        content = self.generate(formatted)
        return AIMessage(content=content)

    @staticmethod
    def _to_openai_messages(messages: Any) -> list[dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        formatted: list[dict[str, str]] = []
        items = messages if isinstance(messages, list) else [messages]
        for m in items:
            if isinstance(m, BaseMessage):
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(m.type, "user")
                formatted.append({"role": role, "content": m.content})
            elif isinstance(m, dict):
                formatted.append(m)
        return formatted
