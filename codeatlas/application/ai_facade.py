"""
CodeAtlas AI Facade
====================

Provides a **single, clean interface** for the Backend team to consume every
AI capability offered by CodeAtlas, without touching internal technical
details (embeddings, vector-DB queries, GPU tensors, prompt engineering, …).

Design decisions
----------------
* **Encapsulation** – callers never import anything from ``infrastructure``,
  ``qdrant_client``, ``langchain``, ``torch``, or ``openai``.  They receive
  plain Pydantic models defined in ``codeatlas.domain``.
* **DDD placement** – lives in the *Application* layer because it orchestrates
  domain objects and infrastructure services to fulfil use-cases.
* **Async-first** – every public method is ``async`` so it can be called from
  FastAPI route handlers without blocking the event loop.
* **Config-driven** – automatically selects the correct LLM backend (Groq /
  Modal+vLLM / Ollama / OpenAI) based on ``codeatlas.settings``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from loguru import logger

from codeatlas.domain.inference import (
    ReasoningStep,
    ReasoningStepType,
    ReasoningStepsResponse,
)
from codeatlas.settings import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REASONING_SYSTEM_PROMPT: str = (
    "You are CodeAtlas — an expert AI reasoning engine.\n\n"
    "Given a user query, decompose it into **exactly four** structured "
    "reasoning steps.  Each step belongs to one of the following phases, "
    "in order:\n\n"
    "1. **UNDERSTAND** – Restate the intent, identify ambiguities, and "
    "determine the knowledge domains involved.\n"
    "2. **PLAN** – Outline the concrete sub-tasks, data sources, or "
    "algorithms needed to answer the query.\n"
    "3. **EXECUTE** – Describe how each sub-task would be carried out "
    "(logic, data transformations, API calls, etc.).\n"
    "4. **VERIFY** – Explain how the answer would be validated for "
    "correctness, completeness, and hallucination risk.\n\n"
    "Return your answer as a **JSON array** with exactly 4 objects.  "
    "Each object MUST have the keys:\n"
    '  `"step_number"` (int 1-4),\n'
    '  `"step_type"` (one of "understand", "plan", "execute", "verify"),\n'
    '  `"title"` (≤120 chars),\n'
    '  `"description"` (detailed, ≤2000 chars),\n'
    '  `"confidence"` (float 0.0–1.0).\n\n'
    "Output ONLY the JSON array — no markdown fences, no commentary."
)

_STEP_TYPE_ORDER: list[ReasoningStepType] = [
    ReasoningStepType.UNDERSTAND,
    ReasoningStepType.PLAN,
    ReasoningStepType.EXECUTE,
    ReasoningStepType.VERIFY,
]


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class CodeAtlasFacade:
    """
    Unified entry-point for all CodeAtlas AI capabilities.

    Usage from the Backend / API layer::

        facade = CodeAtlasFacade()
        result = await facade.extract_reasoning_steps_from_query(
            query="How do I add Redis caching to a FastAPI service?",
            session_id="sess-abc-123",
        )
        print(result.steps)

    The Backend never needs to know *which* LLM provider is in use, how
    prompts are formatted, or how the JSON is parsed.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self._model_id: str = self._resolve_model_id()
        logger.info(f"CodeAtlasFacade initialised  (model_id={self._model_id})")

    # ------------------------------------------------------------------ #
    #  Public API – existing services (thin delegation)
    # ------------------------------------------------------------------ #

    async def run_research_agent(self, query: str) -> None:
        """
        Execute the ReAct research agent and return the structured result.

        Not implemented yet — ``codeatlas.application.agents`` was removed in
        Phase 0.5 (it held only mock logic). Rebuilt in Phase 3 (LangGraph
        mode QA) per CodeAtlas spec §2.4.
        """
        raise NotImplementedError(
            "run_research_agent has no agent to delegate to yet. "
            "Implemented in CodeAtlas roadmap Phase 3 (LangGraph mode QA)."
        )

    async def rag_query(self, query: str, repo_id: str, top_k: int = 3) -> str:
        """
        Run the full Hybrid-RAG pipeline (retrieve → rerank → generate)
        and return the final answer string.

        Parameters
        ----------
        query:
            User question that should be answered using the knowledge base.
        repo_id:
            Repo indexed via `python -m codeatlas.ingest`. Required — there is no
            single "current" repo to fall back to.
        top_k:
            Number of top-ranked context chunks to include.

        Returns
        -------
        str
            The LLM-generated answer grounded in retrieved context.
        """
        from codeatlas.application.rag.models import format_context
        from codeatlas.application.rag.retriever import ContextRetriever

        retriever = ContextRetriever(repo_id=repo_id)

        documents = await asyncio.to_thread(retriever.search, query, top_k)
        context: str = format_context(documents)
        answer: str = await asyncio.to_thread(
            self._call_llm_service, query, context
        )
        return answer

    # ------------------------------------------------------------------ #
    #  Public API – NEW: Reasoning Breakdown
    # ------------------------------------------------------------------ #

    async def extract_reasoning_steps_from_query(
        self,
        query: str,
        session_id: str,
    ) -> ReasoningStepsResponse:
        """
        Take a raw user query and use the configured LLM to decompose it
        into four structured reasoning steps (*Understand → Plan → Execute
        → Verify*) **without any retrieval**.

        This is pure LLM reasoning — no vector DB or knowledge-base lookup
        is involved.

        Parameters
        ----------
        query:
            The raw, natural-language question from the end-user.
        session_id:
            An opaque identifier supplied by the caller for tracing /
            correlation purposes.

        Returns
        -------
        ReasoningStepsResponse
            Fully validated Pydantic model containing the four steps,
            model metadata, and timing information.

        Raises
        ------
        ValueError
            If the LLM output cannot be parsed into valid reasoning steps.
        """
        logger.info(
            f"[Facade] extract_reasoning_steps  session={session_id}  "
            f"query={query[:80]}…"
        )

        start_ns: float = time.perf_counter_ns()

        raw_llm_output: str = await asyncio.to_thread(
            self._invoke_llm_for_reasoning, query
        )

        elapsed_ms: float = (time.perf_counter_ns() - start_ns) / 1_000_000

        steps: list[ReasoningStep] = self._parse_reasoning_steps(raw_llm_output)

        response = ReasoningStepsResponse(
            session_id=session_id,
            original_query=query,
            steps=steps,
            model_id=self._model_id,
            processing_time_ms=round(elapsed_ms, 2),
        )

        logger.success(
            f"[Facade] reasoning breakdown OK  "
            f"session={session_id}  elapsed={elapsed_ms:.1f}ms"
        )
        return response

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_model_id() -> str:
        """Return a human-readable model identifier based on current config.

        Mirrors the selection order in `llm_factory.get_llm()`.
        """
        if settings.MODAL_VLLM_BASE_URL:
            return f"modal-vllm/{settings.MODAL_VLLM_MODEL_ID}"
        if settings.USE_GROQ:
            return f"groq/{settings.GROQ_MODEL_ID}"
        if settings.USE_VLLM:
            return f"vllm/{settings.VLLM_MODEL_ID}"
        if settings.USE_OLLAMA:
            return f"ollama/{settings.OLLAMA_MODEL_ID}"
        return f"openai/{settings.OPENAI_MODEL_ID}"

    # -- LLM invocation ------------------------------------------------ #

    def _invoke_llm_for_reasoning(self, query: str) -> str:
        """
        Build the reasoning prompt and call the correct LLM backend.

        Returns the **raw string** output from the model (expected to be
        a JSON array).
        """
        from langchain.schema import HumanMessage, SystemMessage

        from codeatlas.application.utils.llm_factory import get_llm

        llm = get_llm(temperature=0.2)  # low temp → deterministic reasoning
        messages = [
            SystemMessage(content=_REASONING_SYSTEM_PROMPT),
            HumanMessage(content=f"User query:\n{query}"),
        ]

        response = llm.invoke(messages)

        # LangChain returns AIMessage; vLLM returns AIMessage too via invoke()
        return response.content if hasattr(response, "content") else str(response)

    # -- Parsing ------------------------------------------------------- #

    @staticmethod
    def _parse_reasoning_steps(raw: str) -> list[ReasoningStep]:
        """
        Parse the LLM's raw text into exactly 4 validated ``ReasoningStep``
        objects.  Handles common LLM quirks (markdown fences, trailing
        commas, etc.).
        """
        # Strip possible markdown code fences
        cleaned: str = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(f"[Facade] JSON parsing failed: {exc}\nRaw LLM output:\n{raw}")
            raise ValueError(
                "The LLM did not return valid JSON.  "
                "Consider retrying or switching to a more capable model."
            ) from exc

        if not isinstance(parsed, list) or len(parsed) != 4:
            raise ValueError(
                f"Expected a JSON array of exactly 4 steps, got "
                f"{type(parsed).__name__} with {len(parsed) if isinstance(parsed, list) else 'N/A'} items."
            )

        steps: list[ReasoningStep] = []
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"Step {idx} is not a JSON object: {item!r}")
            steps.append(
                ReasoningStep(
                    step_number=item.get("step_number", idx + 1),
                    step_type=ReasoningStepType(item["step_type"]),
                    title=item["title"],
                    description=item["description"],
                    confidence=float(item.get("confidence", 0.8)),
                )
            )

        return steps

    # -- Existing RAG LLM call (reused from inference_pipeline_api) ---- #

    @staticmethod
    def _call_llm_service(query: str, context: str) -> str:
        """Backend selection lives in `get_llm()` (Groq / Modal+vLLM / Ollama / OpenAI,
        per spec §2.6) — this is just the prompt assembly."""
        from langchain.schema import HumanMessage, SystemMessage

        from codeatlas.application.utils.llm_factory import get_llm

        llm = get_llm()
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Use the following "
                    f"context to answer the user query.\n\nContext:\n{context}"
                )
            ),
            HumanMessage(content=query),
        ]
        return llm.invoke(messages).content
