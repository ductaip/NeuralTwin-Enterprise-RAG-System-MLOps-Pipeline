"""Live PostgresSaver check: run a graph, interrupt it, resume from the checkpoint in
a SEPARATE process-level connection. Import success proves nothing about resume.
"""

import os
import sys
from operator import add
from typing import Annotated, TypedDict

os.environ["LANGGRAPH_CHECKPOINT_BACKEND"] = "postgres"
os.environ["LANGGRAPH_POSTGRES_URI"] = "postgresql://codeatlas:codeatlas@localhost:5433/codeatlas_checkpoints"

from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402 — env vars must be set before import

URI = os.environ["LANGGRAPH_POSTGRES_URI"]
THREAD = {"configurable": {"thread_id": "pg-verify-1"}}


class S(TypedDict):
    steps: Annotated[list[str], add]


def step_a(state: S) -> dict:
    return {"steps": ["a"]}


def step_b(state: S) -> dict:
    return {"steps": ["b"]}


def step_c(state: S) -> dict:
    return {"steps": ["c"]}


def build():
    g = StateGraph(S)
    g.add_node("step_a", step_a)
    g.add_node("step_b", step_b)
    g.add_node("step_c", step_c)
    g.add_edge(START, "step_a")
    g.add_edge("step_a", "step_b")
    g.add_edge("step_b", "step_c")
    g.add_edge("step_c", END)
    return g


phase = sys.argv[1]

if phase == "interrupt":
    with PostgresSaver.from_conn_string(URI) as saver:
        saver.setup()
        # interrupt_before stops execution *before* step_c runs, leaving a checkpoint
        app = build().compile(checkpointer=saver, interrupt_before=["step_c"])
        result = app.invoke({"steps": []}, THREAD)
        print("PHASE1 steps after interrupt:", result["steps"])
        snap = app.get_state(THREAD)
        print("PHASE1 next node pending:", snap.next)

elif phase == "resume":
    # Fresh process, fresh connection — state must come from Postgres, not memory.
    with PostgresSaver.from_conn_string(URI) as saver:
        app = build().compile(checkpointer=saver)
        snap = app.get_state(THREAD)
        print("PHASE2 recovered steps from DB:", snap.values.get("steps"))
        print("PHASE2 next node pending:", snap.next)
        result = app.invoke(None, THREAD)  # None = continue from checkpoint
        print("PHASE2 steps after resume:", result["steps"])
