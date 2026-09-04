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

# A dotted identifier (`APIRouter.add_api_route`) or a single CamelCase/snake_case
# token — the actual symbol name inside a structural question, as opposed to the
# whole sentence around it.
_DOTTED_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_BARE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _extract_symbol_mention(query: str) -> str | None:
    """Pull the likely symbol name out of a structural question.

    `search_symbol` exact-matches `qualified_name`/`name` — passing it the raw
    sentence ("who calls APIRouter.add_api_route") never matches anything and falls
    through to a difflib guess against the whole question, which is nonsense. This
    is the fix for a live-verified miss: the structural pre-check ran, but on the
    wrong input.
    """
    dotted = _DOTTED_IDENTIFIER_RE.findall(query)
    if dotted:
        return max(dotted, key=len)

    stopwords = {"who", "calls", "call", "of", "the", "a", "an", "is", "does", "do", "what"}
    candidates = [
        w for w in _BARE_IDENTIFIER_RE.findall(query) if w.lower() not in stopwords and len(w) > 2
    ]
    return max(candidates, key=len) if candidates else None


def _verify_citations_against_chunks(
    answer: str, chunks: list[RetrievedChunk]
) -> tuple[str, float]:
    """Strip unverifiable citations, using the reranked chunks as the evidence set."""
    from codeatlas.eval.citations import strip_invalid_citations

    evidence = [{"result": {"source": c.source_ref}} for c in chunks]
    cleaned, check = strip_invalid_citations(answer, evidence)
    return cleaned, check.validity_rate


def _generate_with_shrinking_context(query: str, chunks: list[RetrievedChunk]) -> str:
    """Call the LLM with as much context as fits, dropping the lowest-ranked chunk on
    a "too large" 413 and retrying — live-verified failure mode: 5 reranked chunks can
    exceed Groq's 8000 TPM ceiling in one request (`openai/gpt-oss-120b`, "Requested
    11838"). `chunks` is already best-first (post-rerank), so dropping from the tail
    keeps the strongest evidence.
    """
    from codeatlas.application.utils.llm_factory import get_llm
    from codeatlas.domain.exceptions import LLMGenerationError

    remaining = list(chunks)
    while remaining:
        context = format_context(remaining)
        prompt = (
            "You are CodeAtlas. Answer the question using ONLY the context below. "
            "Every claim MUST cite its source as [file.py:12-30]. If the context "
            "does not support an answer, say \"không tìm thấy trong codebase\" — "
            "never invent a citation.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n"
        )
        try:
            response = get_llm(temperature=0).invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except LLMGenerationError as e:
            if "too large" not in str(e).lower() and "413" not in str(e):
                raise
            remaining = remaining[:-1]

    return "Không tìm thấy trong codebase (ngữ cảnh vượt giới hạn token của model)."


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

        mention = _extract_symbol_mention(state["query"])
        if round_ == 0 and mention and _STRUCTURAL_RE.search(state["query"]):
            symbol_hit = tools.search_symbol(mention)
            qn = None
            if symbol_hit.get("results"):
                qn = symbol_hit["results"][0]["qualified_name"]
            elif symbol_hit.get("suggestions"):
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
        tool_hits: list[RetrievedChunk] = []
        for item in state["evidence"]:
            if item.get("round") != current_round or item.get("stage") != "retrieved":
                continue
            if "chunk" in item:
                by_retriever[item["source_retriever"]].append(RetrievedChunk(**item["chunk"]))
            elif item.get("source_retriever") == "tool:impact_analysis":
                # `plan`'s structural pre-check (get_callers/impact_analysis) — without
                # this, its findings sat in the trace for citation-history purposes but
                # never reached ranking. Live-verified miss: "who calls
                # APIRouter.add_api_route" answered "không tìm thấy trong codebase"
                # despite the tool call finding the exact right callers.
                for symbol in item["tool_result"].get("impacted_symbols", [])[:5]:
                    src = symbol["source"]
                    read = tools.read_source(src["file_path"], src["start"], src["end"])
                    if "content" not in read:
                        continue
                    tool_hits.append(
                        RetrievedChunk(
                            content=read["content"],
                            qualified_name=symbol["qualified_name"],
                            file_path=src["file_path"],
                            start_line=src["start"],
                            end_line=src["end"],
                            source="graph",
                            score=1.0 / (symbol["distance"] + 1),
                        )
                    )

        fused = reciprocal_rank_fusion(
            [by_retriever["dense"], by_retriever["sparse"], by_retriever["graph"], tool_hits]
        )

        if tool_hits:
            # Live-verified twice (Phase 2 retrieval eval, and again here): the
            # cross-encoder reranker — trained for prose query/passage relevance —
            # systematically demotes exactly-correct structural hits in favour of test
            # files whose *names* lexically resemble the query. Reproduced concretely:
            # RRF correctly ranked `FastAPI.add_api_route` at position 4 of 35 for "who
            # calls APIRouter.add_api_route", but reranking still dropped every graph
            # hit for `test_router_include_context.*` matches. `tool_hits` exists only
            # because the structural-query heuristic already fired, so trust RRF order
            # here rather than a reranker known to be wrong for this evidence type —
            # this is the "Hybrid + RRF, không rerank" row in spec §3.3 Bảng A.
            reranked = fused[:FINAL_K]
        else:
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
            answer = _generate_with_shrinking_context(state["query"], chunks)
            citations = [c.source_ref | {"qualified_name": c.qualified_name} for c in chunks]
            # Same guard as the ReAct loop: a citation not backed by a retrieved chunk
            # is removed, not merely flagged (CLAUDE.md — never present a fabricated
            # citation as legitimate). The evidence here is the reranked chunk list.
            answer, _validity = _verify_citations_against_chunks(answer, chunks)

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
