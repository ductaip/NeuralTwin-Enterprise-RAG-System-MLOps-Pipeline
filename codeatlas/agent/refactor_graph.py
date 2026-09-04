"""LangGraph mode-Refactor graph — spec §2.3.

impact_analysis (graph) -> generate_patch -> sandbox_apply -> run_tests
   |- pass -> interrupt(): human approve -> commit -> END
   |- fail -> repair -> generate_patch (max 3 loops)
"""

import re
import time
import uuid
import os
import subprocess
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt

from codeatlas.agent.state import AtlasState, initial_state
from codeatlas.agent.tools import AgentTools
from codeatlas.agent.trace import AgentTracer
from codeatlas.application.utils.llm_factory import get_llm
from codeatlas.settings import settings


def _extract_symbol_mention(query: str) -> str | None:
    _DOTTED_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
    _BARE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    dotted = _DOTTED_IDENTIFIER_RE.findall(query)
    if dotted:
        return max(dotted, key=len)
    
    stopwords = {"who", "calls", "call", "of", "the", "a", "an", "is", "does", "do", "what"}
    candidates = [
        w for w in _BARE_IDENTIFIER_RE.findall(query) if w.lower() not in stopwords and len(w) > 2
    ]
    return max(candidates, key=len) if candidates else None


def add_refactor_nodes(graph: StateGraph, tools: AgentTools, tracer: AgentTracer):
    def _trace(node: str, **extra) -> None:
        tracer.query = extra.pop("query", tracer.query)
        tracer.log(node, extra.pop("elapsed_s", 0.0), **extra)

    def impact_analysis(state: AtlasState) -> dict:
        t0 = time.perf_counter()
        mention = _extract_symbol_mention(state["query"])
        affected_tests = []
        impacted_symbols = []
        
        if mention:
            symbol_hit = tools.search_symbol(mention)
            qn = None
            if symbol_hit.get("results"):
                qn = symbol_hit["results"][0]["qualified_name"]
            elif symbol_hit.get("suggestions"):
                qn = symbol_hit["suggestions"][0]
            
            if qn:
                impact = tools.impact_analysis(qn)
                impacted_symbols = [s["qualified_name"] for s in impact.get("impacted_symbols", [])]
                affected_tests = [t["file_path"] for t in impact.get("affected_tests", [])]
        
        _trace("impact_analysis", elapsed_s=time.perf_counter() - t0, tests=len(affected_tests))
        return {"impacted_symbols": impacted_symbols, "affected_tests": affected_tests}

    def generate_patch(state: AtlasState) -> dict:
        t0 = time.perf_counter()
        prompt = (
            f"You are a refactoring agent. Create a unified diff or sed script to implement the user's request.\n"
            f"User Request: {state['query']}\n"
        )
        if state["repair_iteration"] > 0:
            prompt += f"\nPrevious attempt failed with output:\n{state['test_output']}\nFix the error and try again."
            
        llm = get_llm(temperature=0)
        response = llm.invoke([{"role": "user", "content": prompt}])
        patch_text = response.content if hasattr(response, "content") else str(response)
        
        _trace("generate_patch", elapsed_s=time.perf_counter() - t0, patch_len=len(patch_text))
        return {"patch": patch_text}

    def sandbox_apply(state: AtlasState) -> dict:
        t0 = time.perf_counter()
        patch = state["patch"]
        sandbox_dir = Path("/tmp/fastapi_codeatlas")
        if sandbox_dir.exists() and patch:
            patch_file = sandbox_dir / f"patch_{uuid.uuid4().hex[:8]}.diff"
            patch_file.write_text(patch)
            subprocess.run(["patch", "-p1", "-i", str(patch_file)], cwd=str(sandbox_dir), capture_output=True)
            
        _trace("sandbox_apply", elapsed_s=time.perf_counter() - t0)
        return {}

    def run_tests(state: AtlasState) -> dict:
        t0 = time.perf_counter()
        sandbox_dir = Path("/tmp/fastapi_codeatlas")
        if not sandbox_dir.exists():
            return {"test_output": "Sandbox not found.", "repair_iteration": state["repair_iteration"]}
            
        tests_to_run = list(set(state["affected_tests"]))
        if not tests_to_run:
            tests_to_run = ["tests/"]
            
        cmd = ["pytest"] + tests_to_run
        result = subprocess.run(cmd, cwd=str(sandbox_dir), capture_output=True, text=True)
        
        _trace("run_tests", elapsed_s=time.perf_counter() - t0, success=(result.returncode == 0))
        return {"test_output": result.stdout + "\n" + result.stderr}

    def check_tests(state: AtlasState) -> str:
        output = state["test_output"] or ""
        if "FAILED" in output or "failed" in output.lower():
            if state["repair_iteration"] >= 3:
                return "end"
            return "repair"
        return "human_approval"

    def repair(state: AtlasState) -> dict:
        return {"repair_iteration": state["repair_iteration"] + 1}

    def human_approval(state: AtlasState) -> dict:
        approve = interrupt("Vui lòng xem xét bản vá. Bấm OK để commit.")
        return {}

    def commit(state: AtlasState) -> dict:
        sandbox_dir = Path("/tmp/fastapi_codeatlas")
        if sandbox_dir.exists():
            subprocess.run(["git", "commit", "-am", f"Refactor: {state['query']}"], cwd=str(sandbox_dir))
        return {"answer": "Refactor applied and committed.", "citations": []}

    graph.add_node("impact_analysis", impact_analysis)
    graph.add_node("generate_patch", generate_patch)
    graph.add_node("sandbox_apply", sandbox_apply)
    graph.add_node("run_tests", run_tests)
    graph.add_node("repair", repair)
    graph.add_node("human_approval", human_approval)
    graph.add_node("commit", commit)

    graph.add_edge("impact_analysis", "generate_patch")
    graph.add_edge("generate_patch", "sandbox_apply")
    graph.add_edge("sandbox_apply", "run_tests")
    
    graph.add_conditional_edges("run_tests", check_tests, {
        "repair": "repair",
        "human_approval": "human_approval",
        "end": END
    })
    
    graph.add_edge("repair", "generate_patch")
    graph.add_edge("human_approval", "commit")
    graph.add_edge("commit", END)
