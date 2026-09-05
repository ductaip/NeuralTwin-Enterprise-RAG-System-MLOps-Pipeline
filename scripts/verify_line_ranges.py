"""Verify line ranges of suspicious functions in Neo4j vs actual source code files."""
from pathlib import Path
from codeatlas.infrastructure.graph.neo4j_adapter import Neo4jAdapter

a = Neo4jAdapter()

target_qns = [
    "fastapi.sse._check_single_line",
    "fastapi.exceptions.WebSocketException.__init__",
    "fastapi._compat.v2.evaluate_forwardref",
    "fastapi.security.api_key.APIKeyHeader.__init__",
    "fastapi.encoders.jsonable_encoder",
]

print("=== 1. Line ranges in Neo4j ===")
nodes = a.execute_read(
    """
    MATCH (f:Function)
    WHERE f.qualified_name IN $qns
    RETURN f.qualified_name AS qn, f.file_path AS fp, f.start_line AS s, f.end_line AS e
    """,
    {"qns": target_qns}
)

sandbox_base = Path("/home/adminn/.cache/codeatlas-eval/fastapi")

for n in nodes:
    qn = n["qn"]
    fp = n["fp"]
    start = n["s"]
    end = n["e"]
    print(f"\nFunction: {qn}")
    print(f"  Neo4j file_path: {fp}")
    print(f"  Neo4j line range: {start} - {end}")
    
    file_full = sandbox_base / fp
    if file_full.exists():
        lines = file_full.read_text().splitlines()
        print(f"  Total lines in actual file: {len(lines)}")
        
        # Display lines around start and end
        print("  Actual code at Neo4j range:")
        snippet = lines[max(0, start-1):min(len(lines), end)]
        for idx, line in enumerate(snippet, start=start):
            print(f"    L{idx}: {line}")
    else:
        print(f"  File not found at {file_full}")

a.close()
