import asyncio
from typing import AsyncGenerator

import opik
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from opik import opik_context
from pydantic import BaseModel, Field

from codeatlas import settings
from codeatlas.application.rag.models import format_context
from codeatlas.application.rag.retriever import ContextRetriever
from codeatlas.application.utils import misc
from codeatlas.application.utils.llm_factory import get_llm
from codeatlas.infrastructure.monitoring.decorators import track_request_metrics
from codeatlas.infrastructure.monitoring.metrics import metrics_endpoint
from codeatlas.infrastructure.opik_utils import configure_opik
from codeatlas.infrastructure.security.jwt import verify_token
from codeatlas.infrastructure.security.rate_limiter import rate_limit_dependency

configure_opik()

app = FastAPI()

# Register the AI Facade router (reasoning-breakdown, etc.)
from codeatlas.infrastructure.api.facade_controller import router as ai_router  # noqa: E402
from codeatlas.infrastructure.api.agent_controller import router as agent_router  # noqa: E402

app.include_router(ai_router)
app.include_router(agent_router)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The user query.")
    repo_id: str = Field(..., description="Repo indexed via `python -m codeatlas.ingest`.")
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str


@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics.
    """
    return metrics_endpoint()


@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    """Backend selection (Groq / Modal+vLLM / Ollama / OpenAI) lives in `get_llm()`,
    per spec §2.6 — this function no longer decides that itself."""
    from langchain.schema import HumanMessage, SystemMessage

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are CodeAtlas, an assistant answering questions about a codebase. "
                "Ground every claim in the provided context and cite it as "
                "[file.py:12-30]. If the context does not contain the answer, say so "
                "explicitly — never invent a citation.\n\nContext:\n" + (context or "")
            )
        ),
        HumanMessage(content=query),
    ]
    return llm.invoke(messages).content


async def stream_rag(query: str, repo_id: str) -> AsyncGenerator[str, None]:
    """
    Simulates streaming response for the RAG pipeline.
    In a real scenario, this would hook into the LLM's streaming callback.
    """
    # 1. Retrieve
    retriever = ContextRetriever(repo_id=repo_id)
    documents = retriever.search(query, k=3)
    context = format_context(documents)

    # 2. Generate (Simulated Stream)
    full_response = call_llm_service(query, context)

    tokens = full_response.split(" ")
    for token in tokens:
        yield f"{token} "
        await asyncio.sleep(0.05) # Simulate token generation delay


@opik.track
def rag(query: str, repo_id: str) -> str:
    retriever = ContextRetriever(repo_id=repo_id)
    documents = retriever.search(query, k=3)
    context = format_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    return answer


@app.post("/rag", response_model=QueryResponse, dependencies=[Depends(rate_limit_dependency), Depends(verify_token)])
@track_request_metrics
async def rag_endpoint(request: QueryRequest):
    try:
        if request.stream:
            return StreamingResponse(
                stream_rag(request.query, request.repo_id), media_type="text/event-stream"
            )

        answer = rag(query=request.query, repo_id=request.repo_id)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
