"""Phase 2 "KIỂM CHỨNG" step: run sample queries against an indexed repo and print
top-5 per retriever (dense/sparse/graph) plus fused+reranked, per docs/PHASE2_RESULTS.md.

Usage:
    python -m codeatlas.ingest --repo <path> --repo-id fastapi   # index once
    poetry run python scripts/verify_phase2_retrieval.py --repo-id fastapi
"""

from __future__ import annotations

import argparse
import time

from codeatlas.application.rag.retriever import ContextRetriever

DEFAULT_QUERIES = [
    "how does jsonable_encoder convert pydantic models to JSON",
    "where is dependency injection resolved for path operations",
    "who calls APIRouter.add_api_route",
    "how are path parameters validated",
    "what handles CORS middleware configuration",
    "how does FastAPI generate the OpenAPI schema",
    "where is request body parsing implemented",
    "how are background tasks executed after a response",
    "what validates the response model before serialization",
    "how does the TestClient send a request to the app",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    retriever = ContextRetriever(repo_id=args.repo_id)

    for i, query in enumerate(DEFAULT_QUERIES, 1):
        t0 = time.perf_counter()
        trace = retriever.search_with_trace(query, k=args.k)
        elapsed = time.perf_counter() - t0

        print(f"\n{'=' * 90}")
        print(f"[{i}] QUERY: {query}   ({elapsed:.1f}s)")
        print(f"{'=' * 90}")
        print(f"HyDE snippet (first 150 chars): {trace.hypothetical_code[:150]!r}")

        print("\n-- DENSE top-5 --")
        for c in trace.dense[:5]:
            print(f"   {c.score:.3f}  {c.qualified_name}")

        print("-- SPARSE top-5 --")
        for c in trace.sparse[:5]:
            print(f"   {c.score:.3f}  {c.qualified_name}")

        print("-- GRAPH top-5 --")
        for c in trace.graph[:5]:
            print(f"   {c.score:.3f}  {c.qualified_name}")

        print("-- FUSED (RRF k=60) top-5 --")
        for c in trace.fused[:5]:
            print(f"   {c.qualified_name}  (source={c.source})")

        print("-- FINAL (cross-encoder reranked) --")
        for c in trace.final:
            print(f"   {c.qualified_name}  [{c.file_path}:{c.start_line}-{c.end_line}]")


if __name__ == "__main__":
    main()
