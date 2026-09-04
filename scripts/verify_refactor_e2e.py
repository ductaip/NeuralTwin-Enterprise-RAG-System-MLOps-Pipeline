import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from codeatlas.agent.refactor_graph import add_refactor_nodes
from codeatlas.agent.state import AtlasState, initial_state
from codeatlas.agent.tools import AgentTools
from codeatlas.agent.trace import AgentTracer
from codeatlas.settings import settings


def run_e2e_verification():
    print("==========================================================================")
    print(" STEP 3: LIVE END-TO-END REFACTOR GRAPH EXECUTION (Successful Simple Task)")
    print("==========================================================================")

    repo_id = "fastapi"
    sandbox_path = Path(settings.SANDBOX_REPO_PATH)
    print(f"Verified Sandbox Repository Path: {sandbox_path}")
    
    # 1. Initialize Tools & Tracer
    tools = AgentTools(repo_id=repo_id)
    tmp_trace = Path("/tmp/refactor_trace.jsonl")
    tracer = AgentTracer(
        trace_path=tmp_trace,
        orchestrator="langgraph",
        query="Refactor fastapi.applications.FastAPI.setup to rename local variable",
        repo_id=repo_id,
        run_id=str(uuid.uuid4())
    )

    # 2. Build Refactor Subgraph with Entrypoint
    builder = StateGraph(AtlasState)
    add_refactor_nodes(builder, tools, tracer)
    builder.add_edge(START, "impact_analysis")
    
    db_path = Path("/tmp/test_checkpoints.sqlite")
    if db_path.exists():
        db_path.unlink()

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)

        thread_id = f"test_thread_{uuid.uuid4().hex[:6]}"
        config = {"configurable": {"thread_id": thread_id}}

        state_dict = initial_state(
            query="Refactor function fastapi.applications.FastAPI.setup to rename local variable",
            repo_id=repo_id,
            mode="refactor"
        )

        print(f"\n[1] Launching Refactor Graph with Query: {state_dict['query']!r}")
        print(f"    Thread ID: {thread_id}")

        # Stream events/nodes
        events = list(app.stream(state_dict, config=config))
        print("\n--- STREAMED NODE OUTPUTS ---")
        for event in events:
            for node_name, node_output in event.items():
                print(f"\n[NODE COMPLETED]: {node_name}")
                if isinstance(node_output, dict):
                    for k, v in node_output.items():
                        if k == "patch":
                            print(f"  patch (first 200 chars):\n{str(v)[:200]}...")
                        elif k == "test_output":
                            print(f"  test_output (first 300 chars):\n{str(v)[:300]}...")
                        else:
                            print(f"  {k}: {v}")

        # Verify state at interrupt
        state_snapshot = app.get_state(config)
        print(f"\n[INTERRUPT VERIFICATION]: Next node waiting: {state_snapshot.next}")
        print(f"  Tasks at interrupt: {state_snapshot.tasks}")
        if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
            print(f"  Interrupt message: {state_snapshot.tasks[0].interrupts[0].value}")

        # Step 5: Resume execution after interrupt
        print("\n==========================================================================")
        print(" STEP 5: RESUMING GRAPH EXECUTION AFTER INTERRUPT()")
        print("==========================================================================")
        
        resume_events = list(app.stream(Command(resume="OK"), config=config))
        for event in resume_events:
            for node_name, node_output in event.items():
                print(f"\n[NODE COMPLETED AFTER RESUME]: {node_name}")
                print(f"  {node_output}")

        final_snapshot = app.get_state(config)
        print(f"\n[FINAL SNAPSHOT]: Next node: {final_snapshot.next}")

    print("\n==========================================================================")
    print(" STEP 4: PROVOKING TEST FAILURE & REPAIR LOOP ACTIVATION")
    print("==========================================================================")
    
    fake_state = initial_state(
        query="Refactor fastapi.encoders.jsonable_encoder",
        repo_id=repo_id,
        mode="refactor"
    )
    fake_state["test_output"] = "FAILED tests/test_encoders.py::test_jsonable_encoder - AssertionError: expected 1 got 2"
    fake_state["affected_tests"] = ["tests/test_encoders.py"]

    check_builder = StateGraph(AtlasState)
    add_refactor_nodes(check_builder, tools, tracer)
    
    repair_node_fn = check_builder.nodes["repair"].runnable.func if hasattr(check_builder.nodes["repair"], "runnable") else check_builder.nodes["repair"].func
    
    repair_res = repair_node_fn(fake_state)
    print(f"\n[REPAIR NODE OUTPUT]: {repair_res}")
    assert repair_res["repair_iteration"] == 1

    print(f"[REPAIR ITERATION 1 STATE]: repair_iteration={repair_res['repair_iteration']}")
    print("Successfully verified repair loop logic & iteration increment!")

    tools.adapter.close()

if __name__ == "__main__":
    run_e2e_verification()
