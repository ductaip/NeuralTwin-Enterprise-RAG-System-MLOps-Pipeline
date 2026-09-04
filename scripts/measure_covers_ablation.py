import numpy as np
from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter

def measure():
    adapter = Neo4jAdapter()
    repo_id = "fastapi"

    # Total tests in repo
    total_tests_res = adapter.execute_read("MATCH (t:Test {repo_id: $repo_id}) RETURN count(t) as c", {"repo_id": repo_id})
    total_tests = total_tests_res[0]["c"]
    print(f"Total Tests in DB for '{repo_id}': {total_tests}")

    # Specific Target Functions
    target_qns = [
        "fastapi.encoders.jsonable_encoder",
        "fastapi.params.Depends",
        "fastapi.applications.FastAPI.get"
    ]

    print("\n==========================================================")
    print(" 1. TARGET FUNCTIONS IMPACT ANALYSIS (Number of tests pulled)")
    print("==========================================================")
    for qn in target_qns:
        print(f"\nTarget Symbol: {qn}")
        modes = [
            ("TESTS-only", "MATCH (t:Test {repo_id: $repo_id})-[r:TESTS]->(impacted:Function)-[c:CALLS*0..3]->(f {repo_id: $repo_id, qualified_name: $qn}) RETURN count(DISTINCT t) AS c", {}),
            ("COVERS (min_hits=1)", "MATCH (t:Test {repo_id: $repo_id})-[r:COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f {repo_id: $repo_id, qualified_name: $qn}) WHERE r.hits >= 1 RETURN count(DISTINCT t) AS c", {}),
            ("COVERS (min_hits=5)", "MATCH (t:Test {repo_id: $repo_id})-[r:COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f {repo_id: $repo_id, qualified_name: $qn}) WHERE r.hits >= 5 RETURN count(DISTINCT t) AS c", {}),
            ("COVERS (min_hits=10)", "MATCH (t:Test {repo_id: $repo_id})-[r:COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f {repo_id: $repo_id, qualified_name: $qn}) WHERE r.hits >= 10 RETURN count(DISTINCT t) AS c", {}),
            ("UNION (min_hits=1)", "MATCH (t:Test {repo_id: $repo_id})-[r:TESTS|COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f {repo_id: $repo_id, qualified_name: $qn}) WHERE type(r)='TESTS' OR r.hits >= 1 RETURN count(DISTINCT t) AS c", {}),
            ("UNION (min_hits=5)", "MATCH (t:Test {repo_id: $repo_id})-[r:TESTS|COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f {repo_id: $repo_id, qualified_name: $qn}) WHERE type(r)='TESTS' OR r.hits >= 5 RETURN count(DISTINCT t) AS c", {}),
            ("UNION (min_hits=10)", "MATCH (t:Test {repo_id: $repo_id})-[r:TESTS|COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f {repo_id: $repo_id, qualified_name: $qn}) WHERE type(r)='TESTS' OR r.hits >= 10 RETURN count(DISTINCT t) AS c", {}),
        ]
        for name, cypher, extra_params in modes:
            res = adapter.execute_read(cypher, {"repo_id": repo_id, "qn": qn, **extra_params})
            cnt = res[0]["c"] if res else 0
            pct = (cnt / total_tests) * 100
            print(f"  {name:25s}: {cnt:4d} tests ({pct:5.1f}% of suite)")

    print("\n==========================================================")
    print(" 2. SUITE-WIDE PER-FUNCTION IMPACT (Median / P95 / Max)")
    print("==========================================================")

    dist_queries = [
        ("TESTS-only", """
            MATCH (f:Function {repo_id: $repo_id})
            OPTIONAL MATCH (t:Test {repo_id: $repo_id})-[r:TESTS]->(impacted:Function)-[c:CALLS*0..3]->(f)
            WITH f, count(DISTINCT t) AS num_tests
            RETURN percentileCont(num_tests, 0.5) AS median,
                   percentileCont(num_tests, 0.95) AS p95,
                   max(num_tests) AS max_val,
                   avg(num_tests) AS mean_val
        """),
        ("COVERS (min_hits=1)", """
            MATCH (f:Function {repo_id: $repo_id})
            OPTIONAL MATCH (t:Test {repo_id: $repo_id})-[r:COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f)
            WHERE r.hits >= 1
            WITH f, count(DISTINCT t) AS num_tests
            RETURN percentileCont(num_tests, 0.5) AS median,
                   percentileCont(num_tests, 0.95) AS p95,
                   max(num_tests) AS max_val,
                   avg(num_tests) AS mean_val
        """),
        ("COVERS (min_hits=5)", """
            MATCH (f:Function {repo_id: $repo_id})
            OPTIONAL MATCH (t:Test {repo_id: $repo_id})-[r:COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f)
            WHERE r.hits >= 5
            WITH f, count(DISTINCT t) AS num_tests
            RETURN percentileCont(num_tests, 0.5) AS median,
                   percentileCont(num_tests, 0.95) AS p95,
                   max(num_tests) AS max_val,
                   avg(num_tests) AS mean_val
        """),
        ("UNION (min_hits=1)", """
            MATCH (f:Function {repo_id: $repo_id})
            OPTIONAL MATCH (t:Test {repo_id: $repo_id})-[r:TESTS|COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f)
            WHERE type(r) = 'TESTS' OR r.hits >= 1
            WITH f, count(DISTINCT t) AS num_tests
            RETURN percentileCont(num_tests, 0.5) AS median,
                   percentileCont(num_tests, 0.95) AS p95,
                   max(num_tests) AS max_val,
                   avg(num_tests) AS mean_val
        """),
        ("UNION (min_hits=5)", """
            MATCH (f:Function {repo_id: $repo_id})
            OPTIONAL MATCH (t:Test {repo_id: $repo_id})-[r:TESTS|COVERS]->(impacted:Function)-[c:CALLS*0..3]->(f)
            WHERE type(r) = 'TESTS' OR r.hits >= 5
            WITH f, count(DISTINCT t) AS num_tests
            RETURN percentileCont(num_tests, 0.5) AS median,
                   percentileCont(num_tests, 0.95) AS p95,
                   max(num_tests) AS max_val,
                   avg(num_tests) AS mean_val
        """),
    ]

    for mode_name, cypher in dist_queries:
        res = adapter.execute_read(cypher, {"repo_id": repo_id})
        r = res[0]
        med = r["median"]
        p95 = r["p95"]
        max_val = r["max_val"]
        mean_val = r["mean_val"]
        pct_med = (med / total_tests) * 100
        pct_p95 = (p95 / total_tests) * 100
        print(f"\nMode: {mode_name:22s}")
        print(f"  Median tests pulled per function: {med:6.1f} ({pct_med:5.2f}% of suite)")
        print(f"  P95 tests pulled per function:    {p95:6.1f} ({pct_p95:5.2f}% of suite)")
        print(f"  Mean tests pulled per function:   {mean_val:6.1f}")
        print(f"  Max tests pulled per function:    {max_val:6d} ({(max_val/total_tests)*100:5.1f}% of suite)")

    adapter.close()

if __name__ == "__main__":
    measure()
