import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger
from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter


def ingest_coverage(repo_id: str, coverage_json: Path, adapter: Neo4jAdapter):
    with open(coverage_json, "r", encoding="utf-8") as f:
        cov_data = json.load(f)

    if "files" not in cov_data:
        raise ValueError("Invalid coverage.json: 'files' key not found.")

    # Mapping from (test_qualified_name, function_qualified_name) -> hits
    # test_qualified_name in our DB might be like `tests.test_main.test_app` 
    # but coverage context might be `tests/test_main.py::test_app`.
    # We will need to map `pytest` contexts to our `Test.qualified_name`.
    
    # 1. Fetch all functions with line ranges
    logger.info(f"Fetching functions for repo {repo_id}...")
    functions_data = adapter.execute_read(
        """
        MATCH (f:Function {repo_id: $repo_id})
        RETURN f.qualified_name AS qn, f.file_path AS file_path, 
               f.start_line AS start_line, f.end_line AS end_line
        """,
        {"repo_id": repo_id}
    )
    
    # Group by file_path for fast lookup
    funcs_by_file = defaultdict(list)
    for row in functions_data:
        funcs_by_file[row["file_path"]].append({
            "qn": row["qn"],
            "start": row["start_line"],
            "end": row["end_line"]
        })
        
    logger.info(f"Fetching tests for repo {repo_id}...")
    tests_data = adapter.execute_read(
        """
        MATCH (t:Test {repo_id: $repo_id})
        RETURN t.qualified_name AS qn, t.file_path AS file_path, t.name AS name
        """,
        {"repo_id": repo_id}
    )
    
    # Coverage contexts usually look like: `test_fastapi.py::test_get_item`
    # or `tests/test_fastapi.py::test_get_item`.
    # We map (file_path, test_name) -> test_qn
    test_lookup = {}
    for row in tests_data:
        # A test file_path might be "tests/test_fastapi.py", name "test_get_item"
        # We also store just the basename to be resilient
        file_path = row["file_path"]
        basename = Path(file_path).name
        name = row["name"]
        test_lookup[(file_path, name)] = row["qn"]
        test_lookup[(basename, name)] = row["qn"]

    covers_hits = defaultdict(int)

    logger.info("Parsing coverage data...")
    # coverage json with --show-contexts looks like:
    # "files": { "fastapi/routing.py": { "contexts": { "123": ["tests/test_routing.py::test_route"] } } }
    for file_path, file_data in cov_data.get("files", {}).items():
        if "contexts" not in file_data:
            continue
            
        file_funcs = funcs_by_file.get(file_path, [])
        if not file_funcs:
            # Try without leading prefix or match exact suffixes if repo paths differ
            for known_file in funcs_by_file:
                if file_path.endswith(known_file) or known_file.endswith(file_path):
                    file_funcs = funcs_by_file[known_file]
                    break
        if not file_funcs:
            continue
            
        for line_str, contexts in file_data["contexts"].items():
            line_num = int(line_str)
            # Find which function this line belongs to
            func_qn = None
            for func in file_funcs:
                if func["start"] <= line_num <= func["end"]:
                    func_qn = func["qn"]
                    break
                    
            if not func_qn:
                continue
                
            for ctx in contexts:
                # context format: `path/to/test.py::test_name` 
                # or `path/to/test.py::test_name[param1]`
                if "::" not in ctx:
                    continue
                    
                parts = ctx.split("::")
                test_file = parts[0]
                test_name = parts[-1].split("[")[0] # remove parametrization if present
                
                # Try to map context to test_qn
                test_qn = test_lookup.get((test_file, test_name))
                if not test_qn:
                    test_basename = Path(test_file).name
                    test_qn = test_lookup.get((test_basename, test_name))
                
                if test_qn:
                    covers_hits[(test_qn, func_qn)] += 1

    if not covers_hits:
        logger.warning("No COVERS relationships found. Ensure coverage.json was generated with --show-contexts.")
        return

    logger.info(f"Inserting {len(covers_hits)} COVERS relationships...")
    
    # Delete existing COVERS edges for this repo to be idempotent
    adapter.execute_write(
        """
        MATCH (t:Test {repo_id: $repo_id})-[r:COVERS]->(:Function {repo_id: $repo_id})
        DELETE r
        """,
        {"repo_id": repo_id}
    )
    
    # Batch insert
    batch = []
    for (test_qn, func_qn), hits in covers_hits.items():
        batch.append({"test_qn": test_qn, "func_qn": func_qn, "hits": hits})
        
    adapter.execute_write(
        """
        UNWIND $batch AS row
        MATCH (t:Test {repo_id: $repo_id, qualified_name: row.test_qn})
        MATCH (f:Function {repo_id: $repo_id, qualified_name: row.func_qn})
        MERGE (t)-[r:COVERS]->(f)
        SET r.hits = row.hits
        """,
        {"repo_id": repo_id, "batch": batch}
    )
    logger.info("COVERS ingestion complete.")

    # Measure and report sizes
    logger.info("Computing metrics for docs/PHASE4_RESULTS.md...")
    
    # Calculate % of tests that have TESTS, COVERS, UNION edges
    total_tests_result = adapter.execute_read("MATCH (t:Test {repo_id: $repo_id}) RETURN count(t) AS c", {"repo_id": repo_id})
    total_tests = total_tests_result[0]["c"]
    
    if total_tests == 0:
        return
        
    metrics = {}
    for source in ["TESTS", "COVERS", "UNION"]:
        query = f"""
        MATCH (t:Test {{repo_id: $repo_id}})
        {
            "WHERE (t)-[:TESTS]->()" if source == "TESTS"
            else "WHERE (t)-[:COVERS]->()" if source == "COVERS"
            else "WHERE (t)-[:TESTS|COVERS]->()"
        }
        RETURN count(DISTINCT t) AS c
        """
        res = adapter.execute_read(query, {"repo_id": repo_id})
        metrics[source] = res[0]["c"]

    print(f"Total Tests: {total_tests}")
    for source, count in metrics.items():
        print(f"{source}: {count} tests ({count/total_tests*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--coverage-json", required=True)
    args = parser.parse_args()
    
    adapter = Neo4jAdapter()
    ingest_coverage(args.repo_id, Path(args.coverage_json), adapter)
    adapter.close()
