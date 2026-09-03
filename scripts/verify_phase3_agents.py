"""Phase 3 "KIỂM CHỨNG" step: run 5 sample questions through both orchestrators and
print full trace so they can be compared, per docs/codeatlas_roadmap.md Phase 3.

Usage: poetry run python scripts/verify_phase3_agents.py --repo-id fastapi
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

QUESTIONS = [
    "how does jsonable_encoder convert pydantic models to JSON",
    "who calls APIRouter.add_api_route",
    "what does BackgroundTasks.add_task do",
    "where is request body parsing implemented",
    "how does FastAPI generate the OpenAPI schema",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--trace-dir", default=".trace")
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir)

    from codeatlas.agent.custom_react import run_custom_react
    from codeatlas.agent.langgraph_qa import run_langgraph_qa

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 90}\n[{i}] {q}\n{'=' * 90}")

        t0 = time.perf_counter()
        try:
            custom_result = run_custom_react(q, args.repo_id, trace_dir=trace_dir)
            custom_elapsed = time.perf_counter() - t0
            print(f"\n-- CUSTOM ReAct ({custom_elapsed:.1f}s, {custom_result['tool_calls']} tool calls) --")
            print(custom_result["answer"][:500])
        except Exception as e:
            print(f"\n-- CUSTOM ReAct FAILED: {type(e).__name__}: {e}")

        t0 = time.perf_counter()
        try:
            lg_result = run_langgraph_qa(q, args.repo_id, trace_dir=trace_dir)
            lg_elapsed = time.perf_counter() - t0
            print(f"\n-- LangGraph ({lg_elapsed:.1f}s, {len(lg_result['citations'])} citations) --")
            print(lg_result["answer"][:500])
        except Exception as e:
            print(f"\n-- LangGraph FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
