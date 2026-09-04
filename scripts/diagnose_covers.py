"""Diagnose COVERS edges: verify they exist and why per-function queries return 0."""
from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter

a = Neo4jAdapter()

print("=== Q1: Direct COVERS edges TO jsonable_encoder ===")
r1 = a.execute_read(
    'MATCH (t:Test)-[r:COVERS]->(f:Function {qualified_name: $qn}) '
    'RETURN t.qualified_name AS tqn, r.hits AS hits LIMIT 10',
    {"qn": "fastapi.encoders.jsonable_encoder"}
)
for row in r1:
    print(f"  test={row['tqn']}  hits={row['hits']}")
print(f"  Total rows: {len(r1)}")

print("\n=== Q2: COVERS edges FROM jsonable_encoder (wrong direction?) ===")
r2 = a.execute_read(
    'MATCH (f:Function {qualified_name: $qn})-[r:COVERS]->(t:Test) '
    'RETURN t.qualified_name AS tqn, r.hits AS hits LIMIT 10',
    {"qn": "fastapi.encoders.jsonable_encoder"}
)
for row in r2:
    print(f"  test={row['tqn']}  hits={row['hits']}")
print(f"  Total rows: {len(r2)}")

print("\n=== Q3: Total COVERS edge count in whole graph ===")
r3 = a.execute_read('MATCH (:Test)-[r:COVERS]->(:Function) RETURN count(r) AS c', {})
print(f"  Total COVERS edges: {r3[0]['c']}")

print("\n=== Q4: Sample 15 COVERS target functions (qualified_name) ===")
r4 = a.execute_read(
    'MATCH (:Test)-[:COVERS]->(f:Function) '
    'RETURN DISTINCT f.qualified_name AS qn ORDER BY qn LIMIT 15', {}
)
for row in r4:
    print(f"  {row['qn']}")

print("\n=== Q5: Sample 15 COVERS target functions (file_path) ===")
r5 = a.execute_read(
    'MATCH (:Test)-[:COVERS]->(f:Function) '
    'RETURN DISTINCT f.file_path AS fp ORDER BY fp LIMIT 15', {}
)
for row in r5:
    print(f"  {row['fp']}")

print("\n=== Q6: Does jsonable_encoder exist as Function node? ===")
r6 = a.execute_read(
    "MATCH (f:Function) WHERE f.qualified_name CONTAINS 'jsonable_encoder' "
    "RETURN f.qualified_name AS qn, f.file_path AS fp, f.start_line AS s, f.end_line AS e", {}
)
for row in r6:
    print(f"  qn={row['qn']}  fp={row['fp']}  lines={row['s']}-{row['e']}")

print("\n=== Q7: COVERS targets in fastapi.encoders module ===")
r7 = a.execute_read(
    "MATCH (t:Test)-[r:COVERS]->(f:Function) "
    "WHERE f.qualified_name STARTS WITH 'fastapi.encoders' "
    "RETURN f.qualified_name AS fqn, count(DISTINCT t) AS num_tests, sum(r.hits) AS total_hits", {}
)
for row in r7:
    print(f"  func={row['fqn']}  tests={row['num_tests']}  hits={row['total_hits']}")
if not r7:
    print("  (empty)")

print("\n=== Q8: COVERS targets containing 'fastapi.' — sample 20 ===")
r8 = a.execute_read(
    "MATCH (t:Test)-[r:COVERS]->(f:Function) "
    "WHERE f.qualified_name STARTS WITH 'fastapi.' "
    "RETURN DISTINCT f.qualified_name AS fqn LIMIT 20", {}
)
for row in r8:
    print(f"  {row['fqn']}")
if not r8:
    print("  (empty — COVERS targets may use different prefix)")

print("\n=== Q9: Top 10 COVERS targets by test count ===")
r9 = a.execute_read(
    "MATCH (t:Test)-[r:COVERS]->(f:Function) "
    "RETURN f.qualified_name AS fqn, f.file_path AS fp, count(DISTINCT t) AS num_tests "
    "ORDER BY num_tests DESC LIMIT 10", {}
)
for row in r9:
    print(f"  func={row['fqn']}  file={row['fp']}  tests={row['num_tests']}")

print("\n=== Q10: Sample coverage.json context format (from DB) ===")
# Check what test nodes actually connect via COVERS
r10 = a.execute_read(
    "MATCH (t:Test)-[r:COVERS]->(f:Function) "
    "RETURN t.qualified_name AS tqn, t.file_path AS tfp, f.qualified_name AS fqn "
    "LIMIT 5", {}
)
for row in r10:
    print(f"  test={row['tqn']}  test_file={row['tfp']}  func={row['fqn']}")

a.close()
