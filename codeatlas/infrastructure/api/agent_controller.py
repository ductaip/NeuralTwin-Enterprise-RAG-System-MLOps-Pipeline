"""Agent Controller — API Layer (roadmap Phase 3, item 6: SSE streaming).

Exposes both orchestrators over HTTP: a plain JSON endpoint for either, and an SSE
endpoint that streams LangGraph's own per-node transitions (`stream_mode="updates"`)
rather than faking progress client-side.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from codeatlas.infrastructure.security.jwt import verify_token
from codeatlas.infrastructure.security.rate_limiter import rate_limit_dependency

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    repo_id: str = Field(..., description="Repo indexed via `python -m codeatlas.ingest`.")
    orchestrator: Literal["custom", "langgraph"] = "langgraph"


class AgentQueryResponse(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)
    orchestrator: str


@router.post(
    "/query",
    response_model=AgentQueryResponse,
    dependencies=[Depends(rate_limit_dependency), Depends(verify_token)],
)
async def agent_query(payload: AgentQueryRequest) -> AgentQueryResponse:
    import asyncio

    try:
        if payload.orchestrator == "langgraph":
            from codeatlas.agent.langgraph_qa import run_langgraph_qa

            result = await asyncio.to_thread(run_langgraph_qa, payload.query, payload.repo_id)
            return AgentQueryResponse(
                answer=result["answer"], citations=result["citations"], orchestrator="langgraph"
            )
        else:
            from codeatlas.agent.custom_react import run_custom_react

            result = await asyncio.to_thread(run_custom_react, payload.query, payload.repo_id)
            return AgentQueryResponse(answer=result["answer"], citations=[], orchestrator="custom")
    except Exception as exc:
        logger.exception(f"[AgentController] query failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


async def _stream_langgraph(query: str, repo_id: str) -> AsyncGenerator[str, None]:
    """Push one SSE event per LangGraph node transition (`stream_mode="updates"`),
    then a final `event: done` with the answer + citations."""
    from codeatlas.agent.langgraph_qa import build_qa_graph
    from codeatlas.agent.state import initial_state

    graph, _tracer = build_qa_graph(repo_id)
    compiled = graph.compile()

    final_state: dict = {}
    for update in compiled.stream(initial_state(query, repo_id, mode="qa"), stream_mode="updates"):
        for node_name, node_output in update.items():
            # A node that returns no state update (e.g. `router`, which only routes)
            # surfaces here as None rather than {} in this LangGraph version.
            node_output = node_output or {}
            payload = {"node": node_name, "keys": list(node_output.keys())}
            yield f"event: node\ndata: {json.dumps(payload, default=str)}\n\n"
            final_state.update(node_output)

    done_payload = {
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
    }
    yield f"event: done\ndata: {json.dumps(done_payload, default=str)}\n\n"


async def _stream_custom(query: str, repo_id: str) -> AsyncGenerator[str, None]:
    """Custom ReAct has no native step-by-step generator (roadmap ties streaming
    specifically to LangGraph) — stream the finished answer word-by-word, matching
    the existing `stream_rag` pattern in `inference_pipeline_api.py`."""
    import asyncio

    from codeatlas.agent.custom_react import run_custom_react

    result = await asyncio.to_thread(run_custom_react, query, repo_id)
    for word in result["answer"].split(" "):
        yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
        await asyncio.sleep(0.02)
    yield f"event: done\ndata: {json.dumps({'answer': result['answer'], 'citations': []})}\n\n"


@router.get(
    "/query/stream",
    dependencies=[Depends(rate_limit_dependency), Depends(verify_token)],
)
async def agent_query_stream(
    query: str, repo_id: str, orchestrator: Literal["custom", "langgraph"] = "langgraph"
) -> StreamingResponse:
    generator = (
        _stream_langgraph(query, repo_id) if orchestrator == "langgraph" else _stream_custom(query, repo_id)
    )
    return StreamingResponse(generator, media_type="text/event-stream")
