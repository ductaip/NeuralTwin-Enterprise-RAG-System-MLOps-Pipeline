"""
Facade Controller – API Layer
==============================

Exposes the ``CodeAtlasFacade`` capabilities as REST endpoints.

This module is intentionally placed under ``infrastructure/api/`` because
in DDD the HTTP transport is an *infrastructure concern* — it adapts the
outside world to the *application* layer (the facade).

The controller imports **only** the facade and domain contracts.  It has
zero knowledge of LangChain, Qdrant, PyTorch, or any other ML library.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from codeatlas.application.ai_facade import CodeAtlasFacade
from codeatlas.domain.inference import ReasoningStepsResponse
from codeatlas.infrastructure.security.jwt import verify_token
from codeatlas.infrastructure.security.rate_limiter import rate_limit_dependency

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/ai",
    tags=["AI Capabilities"],
)

# ---------------------------------------------------------------------------
# Shared facade instance (lazy, singleton-per-process)
# ---------------------------------------------------------------------------

_facade: CodeAtlasFacade | None = None


def _get_facade() -> CodeAtlasFacade:
    """Lazily instantiate the facade so the module can be imported without
    triggering heavy model loads at import time."""
    global _facade
    if _facade is None:
        _facade = CodeAtlasFacade()
    return _facade


# ---------------------------------------------------------------------------
# Request / Response schemas (transport-layer only)
# ---------------------------------------------------------------------------


class ReasoningBreakdownRequest(BaseModel):
    """Inbound payload for the reasoning-breakdown endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The raw user question to decompose into reasoning steps.",
        examples=["How do I add Redis caching to a FastAPI service?"],
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Opaque session identifier for tracing and correlation.",
        examples=["sess-abc-123"],
    )


# The response model is the domain contract itself — no extra wrapper needed.
# FastAPI will serialise it automatically via Pydantic.


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/reasoning-breakdown",
    response_model=ReasoningStepsResponse,
    status_code=status.HTTP_200_OK,
    summary="Decompose a query into structured reasoning steps",
    description=(
        "Uses the configured LLM to break a user query into four "
        "structured phases: **Understand → Plan → Execute → Verify**.  "
        "This is a pure reasoning operation — no vector-DB retrieval is "
        "involved."
    ),
    dependencies=[Depends(rate_limit_dependency), Depends(verify_token)],
)
async def reasoning_breakdown(
    payload: ReasoningBreakdownRequest,
) -> ReasoningStepsResponse:
    """
    **POST /api/v1/ai/reasoning-breakdown**

    Accepts a user query and returns a structured 4-step reasoning
    decomposition produced by the LLM.
    """
    facade = _get_facade()

    try:
        result: ReasoningStepsResponse = (
            await facade.extract_reasoning_steps_from_query(
                query=payload.query,
                session_id=payload.session_id,
            )
        )
        return result

    except ValueError as exc:
        logger.warning(f"[Controller] reasoning-breakdown parsing error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(f"[Controller] reasoning-breakdown unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating reasoning steps.",
        ) from exc
