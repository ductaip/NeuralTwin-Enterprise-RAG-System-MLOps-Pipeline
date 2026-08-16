"""Contextual retrieval (spec §2.5): before embedding a chunk, generate a one-sentence
description of what it is / where it lives / who calls it, and embed that alongside the
code. Cached by `hash(chunk_content)` so a re-run over the same commit is free.

Cost note: this is one LLM call per chunk. Groq's free tier is 1000 requests/day — a
5000+ chunk repo blows past that in a single ingest run. Spec §2.6 puts this workload on
Modal + vLLM specifically because it's throughput-bound with no rate limit; until that's
deployed (see `scripts/deploy_modal_vllm.py`), this runs against whatever `get_llm()`
resolves to and is meant to be used on a bounded sample, not a full large-repo ingest.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

from loguru import logger

from codeatlas.application.rag.prompt_templates import ContextualEnrichmentTemplate
from codeatlas.infrastructure.llm.cache import cache_key, get_cache


@dataclass
class EnrichmentInput:
    qualified_name: str
    content: str
    graph_context: str
    """One line describing module/parent-class/callers, built from the graph."""


def _extract_text(response) -> str:
    return response.content if hasattr(response, "content") else str(response)


class ContextualEnricher:
    def __init__(self, llm=None, max_workers: int = 4, use_cache: bool = True):
        self._llm = llm
        self.max_workers = max_workers
        self.use_cache = use_cache
        self._cache = get_cache()
        self._template = ContextualEnrichmentTemplate()

    def _get_llm(self):
        if self._llm is None:
            from codeatlas.application.utils.llm_factory import get_llm

            self._llm = get_llm(temperature=0.0)
        return self._llm

    def _enrich_one(self, item: EnrichmentInput) -> str:
        key = cache_key("contextual_enrich", "chunk_context", item.content, 0.0, None)
        if self.use_cache and key in self._cache:
            return self._cache[key]

        prompt = self._template.create_template().format(
            qualified_name=item.qualified_name,
            graph_context=item.graph_context,
            code=item.content,
        )
        response = self._get_llm().invoke(prompt)
        sentence = _extract_text(response).strip()

        if self.use_cache:
            self._cache[key] = sentence
        return sentence

    def enrich_batch(self, items: list[EnrichmentInput]) -> dict[str, str]:
        """Returns {qualified_name: context_sentence}. Concurrent regardless of backend:
        real batching if `get_llm()` resolves to Modal vLLM (server-side continuous
        batching), best-effort parallelism otherwise."""
        results: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_qn = {executor.submit(self._enrich_one, item): item.qualified_name for item in items}
            for future in concurrent.futures.as_completed(future_to_qn):
                qn = future_to_qn[future]
                try:
                    results[qn] = future.result()
                except Exception as e:
                    logger.warning(f"Contextual enrichment failed for {qn}: {e}")
                    results[qn] = ""
        return results
