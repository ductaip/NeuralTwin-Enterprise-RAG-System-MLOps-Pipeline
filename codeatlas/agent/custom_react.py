"""Custom ReAct loop for mode QA — ~80 lines, no framework.

Written new for Phase 3: the NeuralTwin-era ReAct loop was mock (`_mock_solve_loop`,
hardcoded tool output) and was deleted in Phase 0.5, so there is no baseline to keep —
see docs/codeatlas_roadmap.md "Đính chính" #1. This exists specifically so Bảng B (spec
§3.3) has something real to compare LangGraph against: same tools, same retrieval
(`codeatlas/agent/tools.py`), only the orchestration differs.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from codeatlas.agent.harness import ToolCallHarness
from codeatlas.agent.tools import TOOL_NAMES, AgentTools
from codeatlas.agent.trace import AgentTracer

SYSTEM_PROMPT = f"""You are CodeAtlas, answering questions about a codebase using tools.

Tools: {", ".join(TOOL_NAMES)}

Use exactly this format, one step per turn:
Thought: <reasoning>
Action: <tool name>
Action Input: <JSON object of arguments>

When you have enough evidence:
Thought: <reasoning>
Final Answer: <answer, MUST cite evidence as [file.py:12-30]. If no source supports a
claim, say "không tìm thấy trong codebase" instead of guessing.>
"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def run_custom_react(
    query: str, repo_id: str, trace_dir: Path = Path(".trace"), llm=None
) -> dict:
    from codeatlas.application.utils.llm_factory import get_llm

    llm = llm or get_llm(temperature=0.1)
    tools = AgentTools(repo_id)
    dispatch = tools.as_dispatch_table()
    harness = ToolCallHarness()
    tracer = AgentTracer(trace_dir / "custom_react.jsonl", "custom", query, repo_id, str(uuid.uuid4()))

    evidence: list[dict] = []
    transcript = f"{SYSTEM_PROMPT}\n\nQuestion: {query}\n"

    while True:
        with tracer.step("llm_turn") as ctx:
            response = llm.invoke(transcript)
            text = response.content if hasattr(response, "content") else str(response)
            ctx["output"] = text[:500]

        final = _FINAL_RE.search(text)
        if final:
            answer = final.group(1).strip()
            return {"answer": answer, "evidence": evidence, "tool_calls": harness.calls_used}

        action = _ACTION_RE.search(text)
        if not action:
            transcript += f"\n{text}\nObservation: Malformed step — reply with Action/Action Input or Final Answer.\n"
            continue

        tool_name, raw_args = action.group(1).strip(), action.group(2)
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            transcript += f"\n{text}\nObservation: Action Input was not valid JSON: {raw_args!r}\n"
            continue

        if harness.budget_exceeded:
            return _force_answer(llm, query, evidence, harness.calls_used)

        warning = harness.record(tool_name, args)
        if warning:
            transcript += f"\n{text}\nObservation: {warning}\n"
            continue

        if tool_name not in dispatch:
            transcript += f"\n{text}\nObservation: Unknown tool {tool_name!r}. Available: {', '.join(TOOL_NAMES)}\n"
            continue

        t0 = time.perf_counter()
        result = dispatch[tool_name](**args)
        tracer.log_tool_call(tool_name, args, result, time.perf_counter() - t0)
        evidence.append({"tool": tool_name, "args": args, "result": result})
        transcript += f"\n{text}\nObservation: {json.dumps(result, default=str)}\n"


def _force_answer(llm, query: str, evidence: list[dict], tool_calls: int) -> dict:
    """Budget exhausted: answer from evidence gathered so far, never silently fail."""
    context = json.dumps(evidence, default=str)
    prompt = (
        f"Question: {query}\n\nEvidence gathered (tool budget exhausted):\n{context}\n\n"
        "Answer now using only this evidence, citing [file.py:12-30]. If it is not "
        "enough, say so explicitly rather than guessing."
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer.strip(), "evidence": evidence, "tool_calls": tool_calls, "budget_exhausted": True}
