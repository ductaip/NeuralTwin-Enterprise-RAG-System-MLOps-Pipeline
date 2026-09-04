# Phase 4 Results

## COVERS Ingestion Empirical Metrics

Comparing the test-to-function graph relationships on the `fastapi` repository (Total Tests indexed in Neo4j graph: **2,340**).

| Metric | TESTS (AST-only) | COVERS (Coverage-only) | Union (TESTS + COVERS) |
|---|---|---|---|
| **Tests with Edges** | 449 (19.2%) | 1,091 (46.6%) | **1,302 (55.6%)** |
| **Total Relationships** | 449 | 6,633 | **7,082** |
| **Recall Multiplier** | 1.0x (baseline) | 2.43x | **2.90x** |

### Key Observations
1. **HTTP Boundary Isolation**: AST-only static call extraction (`TESTS`) missed 80.8% of tests because test suites interact with endpoints using `TestClient` or HTTP calls, severing direct AST call graphs.
2. **Dynamic Context Reach**: Coverage ingestion with `pytest --cov --cov-context=test` successfully bridged the HTTP boundary, creating 6,633 `COVERS` relationships and covering 46.6% of tests.
3. **Union Synergy**: Combining `TESTS` and `COVERS` achieves 55.6% coverage across 1,302 test functions.

## LangGraph Refactor Mode Verification

The closed-loop refactoring graph (`codeatlas/agent/refactor_graph.py`) has been implemented and integrated into the primary agent router (`codeatlas/agent/langgraph_qa.py`).

### Verification & Bug Fixes
During live verification on the `fastapi` repository:
1. **Impact Analysis Dict Mapping Bug**: Discovered and fixed a bug where `affected_tests` string list of qualified names was being accessed as dict keys (`t["file_path"]`). Updated extraction to use `affected_tests_source` file paths.
2. **Sandbox Path Configurability**: Added `SANDBOX_REPO_PATH` setting to `codeatlas/settings.py` (default: `/tmp/fastapi_codeatlas`) to decouple local clone location from hardcoded paths.
3. **Pytest Runner Resolution**: Updated `run_tests` node to automatically resolve `sandbox_dir/.venv/bin/pytest` or system `pytest`.
4. **Closed-Loop Execution**:
   - `impact_analysis`: Resolves affected tests via Neo4j `TESTS|COVERS` edges.
   - `generate_patch`: LLM produces code modification patch.
   - `sandbox_apply`: Applies diff patch to isolated sandbox repository.
   - `run_tests`: Executes `pytest` targeted strictly at affected test files.
   - `repair`: Automatic iteration (up to 3 rounds) feeding test errors back to LLM on test failures.
   - `human_approval`: Triggers LangGraph `interrupt()` primitive before committing changes.
