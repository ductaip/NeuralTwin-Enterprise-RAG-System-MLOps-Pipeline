import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from codeatlas.agent.refactor_graph import add_refactor_nodes, _extract_symbol_mention
from codeatlas.agent.trace import AgentTracer
from langgraph.graph import StateGraph, START, END


def test_extract_symbol_mention():
    assert _extract_symbol_mention("Refactor jsonable_encoder function") == "jsonable_encoder"
    assert _extract_symbol_mention("Update fastapi.encoders.jsonable_encoder to add default") == "fastapi.encoders.jsonable_encoder"


def test_refactor_graph_nodes(tmp_path):
    mock_tools = MagicMock()
    mock_tools.search_symbol.return_value = {
        "results": [{"qualified_name": "fastapi.encoders.jsonable_encoder"}]
    }
    mock_tools.impact_analysis.return_value = {
        "impacted_symbols": [{"qualified_name": "fastapi.encoders.jsonable_encoder"}],
        "affected_tests": ["tests.test_jsonable_encoder.test_custom"],
        "affected_tests_source": [
            {"qualified_name": "tests.test_jsonable_encoder.test_custom", "file_path": "tests/test_jsonable_encoder.py"}
        ]
    }

    tracer = AgentTracer(
        trace_path=tmp_path / "trace.jsonl",
        orchestrator="langgraph",
        query="Refactor jsonable_encoder function",
        repo_id="fastapi",
        run_id="test_run_1"
    )
    graph = StateGraph(dict)

    add_refactor_nodes(graph, mock_tools, tracer)
    
    # StateNodeSpec contains a Runnable (RunnableCallable or RunnableLambda) in .runnable
    node_spec = graph.nodes["impact_analysis"]
    impact_fn = getattr(node_spec, "runnable", node_spec)

    state = {
        "query": "Refactor jsonable_encoder function",
        "repair_iteration": 0,
        "test_output": "",
        "affected_tests": [],
        "impacted_symbols": [],
    }

    res = impact_fn.invoke(state) if hasattr(impact_fn, "invoke") else impact_fn(state)
    assert res["affected_tests"] == ["tests/test_jsonable_encoder.py"]
    assert res["impacted_symbols"] == ["fastapi.encoders.jsonable_encoder"]
