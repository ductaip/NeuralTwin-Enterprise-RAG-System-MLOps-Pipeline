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
* **Config-driven** – automatically selects the correct LLM backend (Ollama /
  vLLM / OpenAI / Mock) based on ``codeatlas.settings``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING

from loguru import logger

from codeatlas.domain.inference import (
    ReasoningStep,
    ReasoningStepsResponse,
    ReasoningStepType,
)
from codeatlas.settings import settings

if TYPE_CHECKING:
    from codeatlas.application.agents.research_agent import AgentResult


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
        self._mock: bool = settings.MOCK_LLM
        self._model_id: str = self._resolve_model_id()
        logger.info(
            "CodeAtlasFacade initialised  "
            f"(mock={self._mock}, model_id={self._model_id})"
        )

    # ------------------------------------------------------------------ #
    #  Public API – existing services (thin delegation)
    # ------------------------------------------------------------------ #

    async def run_research_agent(self, query: str) -> "AgentResult":
        """
        Execute the ReAct research agent and return the structured result.

        Parameters
        ----------
        query:
            Free-form natural-language research question.

        Returns
        -------
        AgentResult
            Contains the final answer and the full thought-chain.
        """
        from codeatlas.application.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_mock_llm=self._mock)
        return await asyncio.to_thread(agent.solve, query)

    async def rag_query(self, query: str, top_k: int = 3) -> str:
        """
        Run the full Hybrid-RAG pipeline (retrieve → rerank → generate)
        and return the final answer string.

        Parameters
        ----------
        query:
            User question that should be answered using the knowledge base.
        top_k:
            Number of top-ranked context chunks to include.

        Returns
        -------
        str
            The LLM-generated answer grounded in retrieved context.
        """
        import os

        from codeatlas.application.rag.retriever import ContextRetriever
        from codeatlas.domain.embedded_chunks import EmbeddedChunk

        is_mock: bool = os.getenv("MOCK_LLM", "false").lower() == "true" or self._mock
        retriever = ContextRetriever(mock=is_mock)

        documents = await asyncio.to_thread(retriever.search, query, top_k)
        context: str = EmbeddedChunk.to_context(documents)
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
        """Return a human-readable model identifier based on current config."""
        if settings.MOCK_LLM:
            return "mock-llm"
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
        if self._mock:
            return self._mock_reasoning_response(query)

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

    # -- Mock ---------------------------------------------------------- #

    @staticmethod
    def _mock_reasoning_response(query: str) -> str:
        """
        Return a deterministic JSON string so the full pipeline can be
        exercised without a live LLM or API key.
        """
        return json.dumps(
            [
                {
                    "step_number": 1,
                    "step_type": "understand",
                    "title": "Comprehend the User Intent",
                    "description": (
                        f'The user is asking: "{query}". '
                        "First, we identify the core intent, relevant domain "
                        "(e.g., software engineering, data science), and any "
                        "implicit constraints such as language, framework, or "
                        "deployment target."
                    ),
                    "confidence": 0.95,
                },
                {
                    "step_number": 2,
                    "step_type": "plan",
                    "title": "Design a Solution Strategy",
                    "description": (
                        "Break the problem into sub-tasks: "
                        "(a) gather prerequisite knowledge, "
                        "(b) identify the best architectural approach, "
                        "(c) outline implementation steps with concrete "
                        "libraries or APIs, and "
                        "(d) define acceptance criteria for the answer."
                    ),
                    "confidence": 0.90,
                },
                {
                    "step_number": 3,
                    "step_type": "execute",
                    "title": "Produce the Detailed Answer",
                    "description": (
                        "Execute each sub-task: write code snippets where "
                        "applicable, reference authoritative documentation, "
                        "and assemble a coherent, step-by-step explanation "
                        "that directly addresses the user's query."
                    ),
                    "confidence": 0.88,
                },
                {
                    "step_number": 4,
                    "step_type": "verify",
                    "title": "Validate Correctness & Completeness",
                    "description": (
                        "Cross-check the generated answer: verify code "
                        "compiles, confirm facts against known sources, "
                        "ensure no hallucinated libraries or APIs, and "
                        "confirm the response fully answers the original "
                        "query without omissions."
                    ),
                    "confidence": 0.92,
                },
            ],
            indent=2,
        )

    # -- Existing RAG LLM call (reused from inference_pipeline_api) ---- #

    @staticmethod
    def _call_llm_service(query: str, context: str) -> str:
        """Thin wrapper over the existing LLM call logic."""
        import os

        if os.getenv("MOCK_LLM", "false").lower() == "true" and not settings.USE_OLLAMA:
            return (
                "This is a simulated response.  The RAG system retrieved "
                "relevant documents, but the actual LLM call was skipped "
                "to save API costs."
            )

        if settings.USE_OLLAMA:
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

        from codeatlas.model.inference import (
            InferenceExecutor,
            LLMInferenceSagemakerEndpoint,
        )

        llm = LLMInferenceSagemakerEndpoint(
            endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
            inference_component_name=None,
        )
        return InferenceExecutor(llm, query, context).execute()
