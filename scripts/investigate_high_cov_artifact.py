"""Investigate raw coverage contexts in coverage_full.json for sse.py lines 36-39 and WebSocketException.__init__."""
import json

cov_path = "/tmp/fastapi_codeatlas/coverage_full.json"
with open(cov_path) as f:
    cov_data = json.load(f)

print("=== 1. fastapi/sse.py in coverage_full.json ===")
if "fastapi/sse.py" in cov_data["files"]:
    fdata = cov_data["files"]["fastapi/sse.py"]
    print("  executed_lines:", fdata.get("executed_lines"))
    ctx = fdata.get("contexts", {})
    print("  lines in contexts:", sorted([int(k) for k in ctx.keys()]))
    
    # Check lines 36-39
    for line in range(36, 40):
        l_str = str(line)
        if l_str in ctx:
            c_list = ctx[l_str]
            real = [c for c in c_list if c]
            print(f"\n  Line {line} has {len(c_list)} total contexts ({len(real)} non-empty)")
            if real:
                print(f"  Sample non-empty contexts for line {line}:")
                for c in real[:10]:
                    print(f"    '{c}'")

print("\n=== 2. fastapi/exceptions.py in coverage_full.json ===")
if "fastapi/exceptions.py" in cov_data["files"]:
    fdata = cov_data["files"]["fastapi/exceptions.py"]
    ctx = fdata.get("contexts", {})
    for line in range(128, 155):
        l_str = str(line)
        if l_str in ctx:
            real = [c for c in ctx[l_str] if c]
            if real:
                print(f"  Line {line}: {len(real)} non-empty contexts. Sample: {real[0]}")
                break

a = None
