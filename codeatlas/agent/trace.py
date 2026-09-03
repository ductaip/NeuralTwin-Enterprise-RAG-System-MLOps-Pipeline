"""JSONL trace + Prometheus, shared by both orchestrators (spec §2.4, roadmap item 9).

One JSONL line per node/tool step, so a run can be replayed and compared across
orchestrators without re-running anything — this is what the Phase 3 KIỂM CHỨNG step
("in ra trace đầy đủ để tôi so sánh") reads.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from codeatlas.infrastructure.monitoring.metrics import (
    AGENT_NODE_COUNT,
    AGENT_NODE_LATENCY,
    AGENT_TOOL_CALLS,
)

Orchestrator = Literal["custom", "langgraph"]


@dataclass
class AgentTracer:
    trace_path: Path
    orchestrator: Orchestrator
    query: str
    repo_id: str
    run_id: str

    def __post_init__(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, node: str, elapsed_s: float, **extra) -> None:
        AGENT_NODE_COUNT.labels(orchestrator=self.orchestrator, node=node).inc()
        AGENT_NODE_LATENCY.labels(orchestrator=self.orchestrator, node=node).observe(elapsed_s)

        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "orchestrator": self.orchestrator,
            "query": self.query,
            "repo_id": self.repo_id,
            "node": node,
            "elapsed_s": round(elapsed_s, 4),
            **extra,
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def log_tool_call(self, tool_name: str, args: dict, result: dict, elapsed_s: float) -> None:
        AGENT_TOOL_CALLS.labels(orchestrator=self.orchestrator, tool_name=tool_name).inc()
        self.log("tool_call", elapsed_s, tool_name=tool_name, args=args, result=result)

    @contextmanager
    def step(self, node: str, **extra) -> Iterator[dict]:
        """Usage: `with tracer.step("plan") as ctx: ...; ctx["output"] = plan_text`."""
        t0 = time.perf_counter()
        ctx: dict = {}
        error: str | None = None
        try:
            yield ctx
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.log(node, time.perf_counter() - t0, error=error, **ctx, **extra)
