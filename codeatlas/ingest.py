"""CLI: `python -m codeatlas.ingest --repo <url|path> --lang python`."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger

from codeatlas.ingestion.models import CallEdge, SymbolKind, TestEdge
from codeatlas.ingestion.python_parser import PythonParser
from codeatlas.ingestion.repo_loader import load_repo
from codeatlas.ingestion.symbol_resolver import ResolutionResult, SymbolResolver

PARSERS = {"python": PythonParser}
DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9, 1.0)


def _affected_tests_by_threshold(
    call_edges: list[CallEdge],
    test_edges: list[TestEdge],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    max_hops: int = 3,
) -> dict[float, dict[str, float]]:
    """Measure what a confidence threshold actually costs in tests run.

    This mirrors Cypher [3] in memory so the number is available at ingest time rather
    than in Phase 5. The point is to know early whether the recall-first default drags in
    so many tests that the "12 instead of 3000" claim stops holding: if 0.5 selects 400
    tests where 0.9 selects 12, the threshold — not the graph — is the story.
    """
    tests_of: dict[str, set[str]] = defaultdict(set)
    for edge in test_edges:
        tests_of[edge.target_qn].add(edge.test_qn)

    all_tests = {e.test_qn for e in test_edges}
    total_tests = len(all_tests)
    results: dict[float, dict[str, float]] = {}

    for threshold in thresholds:
        callers_of: dict[str, set[str]] = defaultdict(set)
        for edge in call_edges:
            if edge.confidence >= threshold:
                callers_of[edge.callee_qn].add(edge.caller_qn)

        targets = sorted(tests_of)
        sizes: list[int] = []
        for target in targets:
            impacted = {target}
            frontier = {target}
            for _ in range(max_hops):
                nxt: set[str] = set()
                for node in frontier:
                    nxt |= callers_of.get(node, set()) - impacted
                if not nxt:
                    break
                impacted |= nxt
                frontier = nxt

            selected: set[str] = set()
            for node in impacted:
                selected |= tests_of.get(node, set())
            sizes.append(len(selected))

        if not sizes:
            results[threshold] = {
                "functions_measured": 0,
                "median": 0.0,
                "p95": 0.0,
                "max": 0.0,
                "pct_of_suite": 0.0,
                "total_tests": total_tests,
            }
            continue

        ordered = sorted(sizes)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        median = statistics.median(sizes)
        results[threshold] = {
            "functions_measured": len(sizes),
            "median": median,
            "p95": float(p95),
            "max": float(max(sizes)),
            "pct_of_suite": round(100 * median / total_tests, 2) if total_tests else 0.0,
            "total_tests": total_tests,
        }

    return results


def _print_threshold_table(table: dict[float, dict[str, float]]) -> None:
    print("\nChi phí của ngưỡng confidence (|affected_tests| mỗi hàm):")
    print(f"  {'ngưỡng':>8} {'median':>8} {'p95':>8} {'max':>8} {'% suite':>9}")
    for threshold, row in sorted(table.items()):
        print(
            f"  {threshold:>8.2f} {row['median']:>8.1f} {row['p95']:>8.1f} "
            f"{row['max']:>8.1f} {row['pct_of_suite']:>8.2f}%"
        )
    total = next(iter(table.values()), {}).get("total_tests", 0)
    print(f"  (tổng số test trong repo: {int(total)})")


def _print_summary(
    repo_id: str,
    n_files: int,
    symbols: list,
    result: ResolutionResult,
    elapsed: float,
) -> None:
    functions = [s for s in symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)]
    classes = [s for s in symbols if s.kind is SymbolKind.CLASS]
    tests = [s for s in functions if s.is_test]
    report = result.report

    print(f"\n=== CodeAtlas ingest: {repo_id} ===")
    print(f"  file            {n_files}")
    print(f"  function/method {len(functions)}")
    print(f"  class           {len(classes)}")
    print(f"  test            {len(tests)}")
    print(f"  call site       {report.total_call_sites}")
    print(f"    internal      {report.internal}")
    print(f"    external      {report.external}")
    print(f"    unresolved    {report.unresolved}")
    print(f"  call edge       {len(result.call_edges)}")
    print(f"  inherits edge   {len(result.inheritance_edges)}")
    print(f"  tests edge      {len(result.test_edges)}")
    print(
        f"  resolve rate    {report.internal_resolve_rate:.1%} "
        f"(nội bộ; mẫu số = internal + unresolved = {report.internal_call_sites})"
    )
    print(f"  thời gian       {elapsed:.1f}s")

    print("\n  Top pattern thất bại (chỉ tính unresolved):")
    for reason, count in report.top_patterns(20):
        share = 100 * count / report.unresolved if report.unresolved else 0
        print(f"    {reason:<34} {count:>7}  ({share:.1f}% số unresolved)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codeatlas.ingest")
    parser.add_argument("--repo", required=True, help="Git URL or local path")
    parser.add_argument("--lang", default="python", choices=sorted(PARSERS))
    parser.add_argument("--commit", default=None, help="Checkout this commit before indexing")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--no-write", action="store_true", help="Skip Neo4j; analyse only")
    parser.add_argument("--no-qdrant", action="store_true", help="Skip embedding chunks into Qdrant")
    parser.add_argument(
        "--contextual",
        action="store_true",
        help=(
            "Enable contextual retrieval (LLM-generated context sentence per chunk, "
            "spec §2.5). OFF by default: this is one LLM call per chunk, and Groq's "
            "free tier (1000 req/day) is smaller than most repos' chunk count. Deploy "
            "scripts/deploy_modal_vllm.py and set MODAL_VLLM_BASE_URL first for "
            "full-repo runs."
        ),
    )
    parser.add_argument("--report-dir", default=".", help="Where unresolved_report.json goes")
    parser.add_argument("--max-hops", type=int, default=3)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    loaded = load_repo(args.repo, commit=args.commit, repo_id=args.repo_id)

    try:
        parser_impl = PARSERS[args.lang]()
        module_pairs = []
        for source in loaded.files:
            if parser_impl.can_parse(source.path):
                module_pairs.append((source, parser_impl.parse(source)))
        modules = [parsed for _source, parsed in module_pairs]

        failed = [m for m in modules if m.parse_error]
        if failed:
            logger.warning(f"{len(failed)} file(s) failed to parse; they are excluded.")

        resolver = SymbolResolver(modules)
        result = resolver.resolve_all()
        symbols = list(resolver.symbol_table.values())
        elapsed = time.perf_counter() - started

        _print_summary(loaded.repo_id, len(loaded.files), symbols, result, elapsed)

        threshold_table = _affected_tests_by_threshold(
            result.call_edges, result.test_edges, max_hops=args.max_hops
        )
        _print_threshold_table(threshold_table)

        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "unresolved_report.json"
        payload = result.report.to_dict()
        payload["repo_id"] = loaded.repo_id
        payload["commit_sha"] = loaded.commit_sha
        payload["files_parsed"] = len(modules)
        payload["files_failed"] = len(failed)
        payload["threshold_cost"] = {str(k): v for k, v in threshold_table.items()}
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  báo cáo -> {report_path}")

        if not args.no_write:
            from codeatlas.ingestion.graph_builder import GraphBuilder

            imports = [
                {"file_path": imp.file_path, "module": imp.module, "alias": imp.alias}
                for module in modules
                for imp in module.imports
                if imp.module and imp.symbol != "*"
            ]
            builder = GraphBuilder()
            builder.build(
                repo_id=loaded.repo_id,
                files=loaded.files,
                symbols=symbols,
                call_edges=result.call_edges,
                inheritance_edges=result.inheritance_edges,
                test_edges=result.test_edges,
                imports=imports,
                url=args.repo if args.repo.startswith("http") else "",
                commit_sha=loaded.commit_sha,
                language=args.lang,
            )

        if not args.no_qdrant:
            from codeatlas.ingestion.qdrant_writer import QdrantChunkWriter

            good_pairs = [(s, p) for s, p in module_pairs if not p.parse_error]
            symbols_by_qn = {s.qualified_name: s for s in symbols}

            enricher = None
            if args.contextual:
                from codeatlas.application.rag.contextual_enrichment import ContextualEnricher

                enricher = ContextualEnricher()

            qdrant_started = time.perf_counter()
            writer = QdrantChunkWriter(enricher=enricher)
            qdrant_stats = writer.write(
                repo_id=loaded.repo_id,
                commit_sha=loaded.commit_sha,
                language=args.lang,
                modules=good_pairs,
                symbols_by_qn=symbols_by_qn,
            )
            qdrant_elapsed = time.perf_counter() - qdrant_started
            print(
                f"  Qdrant          {qdrant_stats.chunks_written} chunk "
                f"({qdrant_elapsed:.1f}s)"
            )
    finally:
        if loaded.cleanup_dir and loaded.cleanup_dir.exists():
            shutil.rmtree(loaded.cleanup_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
