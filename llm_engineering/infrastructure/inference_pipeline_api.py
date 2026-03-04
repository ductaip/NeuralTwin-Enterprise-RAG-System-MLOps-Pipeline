import asyncio
import os
import time
from typing import AsyncGenerator

import opik
from loguru import logger
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, Response
from opik import opik_context
from pydantic import BaseModel, Field

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint
from llm_engineering.infrastructure.security.jwt import verify_token
from llm_engineering.infrastructure.security.rate_limiter import rate_limit_dependency
from llm_engineering.infrastructure.monitoring.decorators import track_request_metrics
from llm_engineering.infrastructure.monitoring.metrics import metrics_endpoint

configure_opik()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The user query.")
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
    # MOCK MODE FOR PORTFOLIO
    if os.getenv("MOCK_LLM", "false").lower() == "true" and not settings.USE_OLLAMA:
        logger.info("Using MOCK LLM response.")
        return "This is a simulated response. The RAG system retrieved relevant documents, but the actual LLM call was skipped to save AWS costs. In a production environment, this would call Llama 3 on SageMaker."

    if settings.USE_OLLAMA:
        from llm_engineering.application.utils.llm_factory import get_llm
        from langchain.schema import HumanMessage, SystemMessage
        
        logger.info(f"Using Ollama Model: {settings.OLLAMA_MODEL_ID}")
        llm = get_llm()
        
        # Simple invoke for local testing
        messages = [
            SystemMessage(content=f"You are a helpful assistant. Use the following context to answer the user query.\n\nContext:\n{context}"),
            HumanMessage(content=query)
        ]
        return llm.invoke(messages).content

    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer

async def stream_rag(query: str) -> AsyncGenerator[str, None]:
    """
    Simulates streaming response for the RAG pipeline.
    In a real scenario, this would hook into the LLM's streaming callback.
    """
    is_mock = os.getenv("MOCK_LLM", "false").lower() == "true"
    # 1. Retrieve
    retriever = ContextRetriever(mock=is_mock)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)
    
    # 2. Generate (Simulated Stream)
    full_response = call_llm_service(query, context)
    
    tokens = full_response.split(" ")
    for token in tokens:
        yield f"{token} "
        await asyncio.sleep(0.05) # Simulate token generation delay


@opik.track
def rag(query: str) -> str:
    is_mock = os.getenv("MOCK_LLM", "false").lower() == "true"
    retriever = ContextRetriever(mock=is_mock)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

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
            return StreamingResponse(stream_rag(request.query), media_type="text/event-stream")
            
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
