"""Client for the vLLM server deployed by `scripts/deploy_modal_vllm.py`.

Eval/ablation backend per spec §2.6: no rate limit, so the client's job is throughput
(batch inference) and reproducibility (fixed seed, disk cache), not backoff.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from loguru import logger
from openai import OpenAI

from codeatlas.domain.exceptions import LLMGenerationError
from codeatlas.infrastructure.llm.cache import cache_key, get_cache
from codeatlas.settings import settings


class ModalVLLMProvider:
    """Duck-typed like `GroqProvider`/`VLLMClient`: `.invoke()` for LCEL chains, plus
    `.generate()` / `.generate_batch()` for direct use (contextual retrieval, eval).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        seed: int = 0,
        use_cache: bool = True,
        max_workers: int = 8,
    ):
        self.base_url = base_url or settings.MODAL_VLLM_BASE_URL
        if not self.base_url:
            raise LLMGenerationError(
                "MODAL_VLLM_BASE_URL is not set. Deploy scripts/deploy_modal_vllm.py "
                "and set the endpoint URL in .env, or pass base_url explicitly."
            )
        self.model = model or settings.MODAL_VLLM_MODEL_ID
        self.seed = seed
        self.use_cache = use_cache
        self.max_workers = max_workers
        self._client = OpenAI(api_key=settings.MODAL_API_KEY or "EMPTY", base_url=self.base_url)
        self._cache = get_cache()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        prompt_for_cache = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        key = cache_key("modal_vllm", self.model, prompt_for_cache, temperature, self.seed)
        if self.use_cache and key in self._cache:
            return self._cache[key]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=self.seed,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            logger.exception(f"Modal vLLM generation failed: {e}")
            raise LLMGenerationError(f"Modal vLLM generation failed: {e}") from e

        if self.use_cache:
            self._cache[key] = content
        return content

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        """Fire many requests concurrently; the server does continuous batching."""

        def _one(prompt: str) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return self.generate(messages, max_tokens=max_tokens, temperature=temperature)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(_one, prompts))

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
