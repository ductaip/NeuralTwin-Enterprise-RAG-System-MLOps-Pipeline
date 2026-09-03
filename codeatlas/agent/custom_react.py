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
from codeatlas.domain.exceptions import LLMGenerationError

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
_CITATION_RE = re.compile(r"\[([\w./\\-]+\.\w+):\d+-\d+\]")


def _known_file_paths(evidence: list[dict]) -> set[str]:
    """Every `file_path` that appeared in a real tool result, regardless of which
    tool's shape it came from (`source`, `results[].source`, `callers[].source`, ...)."""
    paths: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            fp = node.get("file_path")
            if isinstance(fp, str):
                paths.add(fp)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for item in evidence:
        walk(item.get("result"))
    return paths


def _flag_unverifiable_citations(answer: str, evidence: list[dict]) -> str:
    """Hard rule (CLAUDE.md): never present a fabricated citation as legitimate.
    Live-verified failure: the model cited `src/http/request_parser.py:45-78` — a file
    that does not exist anywhere in the indexed repo — as if it were real evidence.
    This is a mechanical last-line check, not a fix for hallucination in general: any
    `[file.py:N-M]` whose file never appeared in a real tool result gets flagged
    in-line rather than left to read as a trustworthy source.
    """
    known = _known_file_paths(evidence)
    unverifiable = {m for m in _CITATION_RE.findall(answer) if m not in known}
    if not unverifiable:
        return answer
    files = ", ".join(sorted(unverifiable))
    return (
        f"{answer}\n\n"
        f"[CẢNH BÁO: trích dẫn tới {files} không khớp với bất kỳ tool result nào đã "
        f"thu thập trong phiên này — có thể là bịa đặt, đừng tin mà không kiểm tra lại.]"
    )


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

    malformed_retries = 0

    while True:
        with tracer.step("llm_turn") as ctx:
            try:
                response = llm.invoke(transcript)
            except LLMGenerationError as e:
                # Live-verified: some Groq-hosted models (e.g. openai/gpt-oss-120b)
                # spontaneously emit a native tool-call even though this loop never
                # declares a `tools` schema — Groq's server then 400s with "Tool
                # choice is none, but model called a tool" instead of returning text.
                # Recoverable: tell the model plain text is required and retry, same
                # as any other malformed step, rather than crashing the whole run.
                malformed_retries += 1
                # NOT "error" — `tracer.step()`'s own `finally` already passes
                # `error=...` to `AgentTracer.log()`; reusing the name here collided
                # with it (`TypeError: got multiple values for keyword argument
                # 'error'`), which crashed the loop live on 2 of 5 verification
                # queries. `ctx` is spread as `**ctx` into that same call, so any key
                # here must avoid `log()`'s own parameter names.
                ctx["llm_error"] = str(e)
                if malformed_retries > 3:
                    return _force_answer(llm, query, evidence, harness.calls_used)
                transcript += (
                    "\nObservation: Your last step was rejected by the API "
                    "(attempted a native tool call). Respond in plain text only, "
                    "using the Thought/Action/Action Input or Thought/Final Answer "
                    "format described above — never call a tool through any other "
                    "mechanism.\n"
                )
                continue
            text = response.content if hasattr(response, "content") else str(response)
            ctx["output"] = text[:500]

        final = _FINAL_RE.search(text)
        if final:
            answer = _flag_unverifiable_citations(final.group(1).strip(), evidence)
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
        try:
            result = dispatch[tool_name](**args)
        except TypeError as e:
            # Wrong/unknown argument names are a common, recoverable LLM tool-call
            # mistake — surface it as an observation so the model can self-correct
            # instead of crashing the whole run.
            transcript += f"\n{text}\nObservation: Bad arguments for {tool_name}: {e}\n"
            continue
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
    answer = _flag_unverifiable_citations(answer.strip(), evidence)
    return {"answer": answer, "evidence": evidence, "tool_calls": tool_calls, "budget_exhausted": True}
