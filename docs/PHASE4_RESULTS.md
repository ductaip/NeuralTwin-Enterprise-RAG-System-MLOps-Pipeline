# Phase 4 Results

## COVERS Ingestion Metrics

Comparing the test-to-function graph relationships on the `fastapi` repository.

| Metric | TESTS (AST-only) | COVERS (Coverage-only) | Union (TESTS + COVERS) |
|---|---|---|---|
| Tests with edges | TBD | TBD | TBD |
| Median functions per test | TBD | TBD | TBD |
| Median tests per function | TBD | TBD | TBD |

**Note:** The exact numbers are pending a complete successful run of the `fastapi` test suite using `pytest --cov=fastapi --cov-context=test`. Currently, running the suite locally requires full dependency resolution (e.g. via `uv sync` or fixing `pytest` dependencies).

## LangGraph Refactor Mode

The refactoring graph is implemented and integrated into the primary agent router.
The flow relies on `COVERS` to ensure robust impact analysis before applying a patch to the sandbox environment and executing the targeted tests.
