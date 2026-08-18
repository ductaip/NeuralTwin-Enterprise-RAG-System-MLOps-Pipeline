"""LangGraph mode-QA graph — spec §2.3.

START -> router -> plan -> [retrieve_dense, retrieve_sparse, retrieve_graph] (fan-out)
      -> fuse_rerank -> [enough_evidence?] -> plan (loop, max 2 rounds) | generate -> END

`retrieve_*` are three separate nodes with edges fanning out from `plan` and back into
`fuse_rerank` — LangGraph's own parallel-branch execution, not the Python-thread fan-out
`ContextRetriever` already does internally in Phase 2. That duplication is deliberate:
the roadmap asks specifically for the *graph* to fan out, so this is the one place
Phase 3 doesn't just call the Phase 2 retriever wholesale — it reuses the same
`DenseCodeRetriever`/`SparseCodeRetriever`/`GraphRetriever`/`Reranker`/`HydeGenerator`
building blocks underneath, at the granularity LangGraph's parallelism can act on.

`tool_budget_used` is reused as the retrieval-round counter (AtlasState is fixed by
spec — no new fields). `get_callers`/`impact_analysis`/etc. are invoked from `plan` via
the same `AgentTools` the custom ReAct loop uses, gated by a structural-query heuristic,
so both orchestrators share tools without this graph re-implementing free-form ReAct.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from codeatlas.agent.state import AtlasState, initial_state
from codeatlas.agent.tools import AgentTools
from codeatlas.agent.trace import AgentTracer
from codeatlas.application.rag.dense_retriever import DenseCodeRetriever
from codeatlas.application.rag.graph_retriever import GraphRetriever
from codeatlas.application.rag.hyde_generator import HydeGenerator
from codeatlas.application.rag.models import RetrievedChunk, format_context
from codeatlas.application.rag.reranking import Reranker
from codeatlas.application.rag.rrf import reciprocal_rank_fusion
from codeatlas.application.rag.sparse_retriever import SparseCodeRetriever
from codeatlas.domain.queries import Query

MAX_RETRIEVAL_ROUNDS = 2
FETCH_K = 15
FINAL_K = 5

_STRUCTURAL_RE = re.compile(
    r"\b(who calls?|callers? of|callees? of|who uses?|impact of|affected by|"
    r"gọi h[àa]m n[àa]y|ai gọi|ảnh hưởng)\b",
    re.IGNORECASE,
)


def build_qa_graph(repo_id: str, trace_dir: Path = Path(".trace"), run_id: str | None = None):
    tools = AgentTools(repo_id)
    tracer = AgentTracer(
        trace_dir / "langgraph_qa.jsonl", "langgraph", "", repo_id, run_id or str(uuid.uuid4())
    )
    hyde = HydeGenerator()
    dense = DenseCodeRetriever(repo_id)
    sparse = SparseCodeRetriever(repo_id)
    graph_r = GraphRetriever(repo_id)
    reranker = Reranker()

    def _trace(node: str, **extra) -> None:
        tracer.query = extra.pop("query", tracer.query)
        tracer.log(node, extra.pop("elapsed_s", 0.0), **extra)

    def router(state: AtlasState) -> dict:
        return {}

    def route_by_mode(state: AtlasState) -> str:
        return "planning" if state["mode"] == "qa" else "refactor_not_implemented"

    def refactor_not_implemented(state: AtlasState) -> dict:
        return {"answer": "Mode refactor chưa implement — xem CodeAtlas roadmap Phase 4."}

    def plan(state: AtlasState) -> dict:
        t0 = time.perf_counter()
        round_ = state["tool_budget_used"]
        structural_evidence: list[dict] = []

        if round_ == 0 and _STRUCTURAL_RE.search(state["query"]):
            symbol_hit = tools.search_symbol(state["query"])
            qn = None
            if "results" in symbol_hit and symbol_hit["results"]:
                qn = symbol_hit["results"][0]["qualified_name"]
            elif "suggestions" in symbol_hit and symbol_hit["suggestions"]:
                qn = symbol_hit["suggestions"][0]
            if qn:
                impact = tools.impact_analysis(qn)
                structural_evidence.append(
                    {"stage": "retrieved", "round": round_, "source_retriever": "tool:impact_analysis", "tool_result": impact}
                )

        # Round 0: HyDE-augmented dense search. Retry round: literal query — if the
        # hypothetical snippet didn't find enough, a different strategy is worth more
        # than repeating the same one and hoping.
        hypothetical = hyde.generate(state["query"]) if round_ == 0 else state["query"]
        _trace("plan", elapsed_s=time.perf_counter() - t0, round=round_, plan_preview=hypothetical[:200])
        return {"plan": hypothetical, "evidence": structural_evidence, "tool_budget_used": round_ + 1}

    def _retrieve_node(name: str, fn) -> callable:
        def node(state: AtlasState) -> dict:
            t0 = time.perf_counter()
            query = state["plan"] if name == "dense" else state["query"]
            chunks: list[RetrievedChunk] = fn(query, FETCH_K)
            _trace(f"retrieve_{name}", elapsed_s=time.perf_counter() - t0, hits=len(chunks))
            return {
                "evidence": [
                    {
                        "stage": "retrieved",
                        "round": state["tool_budget_used"] - 1,
                        "source_retriever": name,
                        "chunk": c.model_dump(),
                    }
                    for c in chunks
                ]
            }

        return node

    retrieve_dense = _retrieve_node("dense", dense.search)
    retrieve_sparse = _retrieve_node("sparse", sparse.search)
    retrieve_graph = _retrieve_node("graph", graph_r.search)

    def fuse_rerank(state: AtlasState) -> dict:
        t0 = time.perf_counter()
        current_round = state["tool_budget_used"] - 1
        by_retriever: dict[str, list[RetrievedChunk]] = {"dense": [], "sparse": [], "graph": []}
        for item in state["evidence"]:
            if item.get("stage") == "retrieved" and item.get("round") == current_round and "chunk" in item:
                by_retriever[item["source_retriever"]].append(RetrievedChunk(**item["chunk"]))

        fused = reciprocal_rank_fusion([by_retriever["dense"], by_retriever["sparse"], by_retriever["graph"]])
        reranked = (
            reranker.generate(query=Query.from_str(state["query"]), chunks=fused, keep_top_k=FINAL_K)
            if fused
            else []
        )
        _trace("fuse_rerank", elapsed_s=time.perf_counter() - t0, round=current_round, fused=len(fused), final=len(reranked))
        return {
            "evidence": [
                {"stage": "fused", "round": current_round, "chunks": [c.model_dump() for c in reranked]}
            ]
        }

    def enough_evidence(state: AtlasState) -> str:
        latest_fused = [e for e in state["evidence"] if e.get("stage") == "fused"]
        chunk_count = len(latest_fused[-1]["chunks"]) if latest_fused else 0
        if chunk_count > 0 or state["tool_budget_used"] >= MAX_RETRIEVAL_ROUNDS:
            return "generate"
        return "planning"

    def generate(state: AtlasState) -> dict:
        t0 = time.perf_counter()
        latest_fused = [e for e in state["evidence"] if e.get("stage") == "fused"]
        chunks = [RetrievedChunk(**c) for c in (latest_fused[-1]["chunks"] if latest_fused else [])]

        if not chunks:
            answer = "Không tìm thấy trong codebase."
            citations: list[dict] = []
        else:
            from codeatlas.application.utils.llm_factory import get_llm

            context = format_context(chunks)
            prompt = (
                "You are CodeAtlas. Answer the question using ONLY the context below. "
                "Every claim MUST cite its source as [file.py:12-30]. If the context "
                "does not support an answer, say \"không tìm thấy trong codebase\" — "
                "never invent a citation.\n\n"
                f"Context:\n{context}\n\nQuestion: {state['query']}\n"
            )
            response = get_llm(temperature=0).invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            citations = [c.source_ref | {"qualified_name": c.qualified_name} for c in chunks]

        _trace("generate", elapsed_s=time.perf_counter() - t0, answer_len=len(answer))
        return {"answer": answer.strip(), "citations": citations}

    # Node id "planning" (not "plan") — LangGraph 0.4.5's StateGraph.add_node rejects a
    # node name that collides with a state channel key, and `plan` is a fixed AtlasState
    # field per spec §2.3.
    graph = StateGraph(AtlasState)
    graph.add_node("router", router)
    graph.add_node("planning", plan)
    graph.add_node("retrieve_dense", retrieve_dense)
    graph.add_node("retrieve_sparse", retrieve_sparse)
    graph.add_node("retrieve_graph", retrieve_graph)
    graph.add_node("fuse_rerank", fuse_rerank)
    graph.add_node("generate", generate)
    graph.add_node("refactor_not_implemented", refactor_not_implemented)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_by_mode, {"planning": "planning", "refactor_not_implemented": "refactor_not_implemented"})
    graph.add_edge("refactor_not_implemented", END)
    graph.add_edge("planning", "retrieve_dense")
    graph.add_edge("planning", "retrieve_sparse")
    graph.add_edge("planning", "retrieve_graph")
    graph.add_edge("retrieve_dense", "fuse_rerank")
    graph.add_edge("retrieve_sparse", "fuse_rerank")
    graph.add_edge("retrieve_graph", "fuse_rerank")
    graph.add_conditional_edges("fuse_rerank", enough_evidence, {"planning": "planning", "generate": "generate"})
    graph.add_edge("generate", END)

    return graph, tracer


def run_langgraph_qa(query: str, repo_id: str, trace_dir: Path = Path(".trace")) -> dict:
    graph, tracer = build_qa_graph(repo_id, trace_dir)
    compiled = graph.compile()
    result = compiled.invoke(initial_state(query, repo_id, mode="qa"))
    return {"answer": result["answer"], "citations": result["citations"], "evidence": result["evidence"]}
