"""Check coverage.json for fastapi/encoders.py presence and context format."""
import json
import sys

cov_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fastapi_codeatlas/coverage.json"

with open(cov_path) as f:
    d = json.load(f)

files = list(d.get("files", {}).keys())
print(f"Total source files in coverage.json: {len(files)}")

print("\nFiles containing 'encoder':")
for fp in sorted(files):
    if "encoder" in fp.lower():
        print(f"  {fp}")

print("\nSample of first 20 file paths:")
for fp in sorted(files)[:20]:
    print(f"  {fp}")

# Check if fastapi/encoders.py is there
target = "fastapi/encoders.py"
if target in d["files"]:
    fdata = d["files"][target]
    print(f"\n{target} FOUND in coverage.json")
    print(f"  has 'contexts': {'contexts' in fdata}")
    if "contexts" in fdata:
        ctx = fdata["contexts"]
        print(f"  num lines with contexts: {len(ctx)}")
        for i, (line, contexts) in enumerate(sorted(ctx.items(), key=lambda x: int(x[0]))):
            if i < 8:
                sample = contexts[:3]
                suffix = f"... (+{len(contexts)-3} more)" if len(contexts) > 3 else ""
                print(f"  line {line}: {sample}{suffix}")
    if "executed_lines" in fdata:
        print(f"  executed_lines count: {len(fdata['executed_lines'])}")
    if "missing_lines" in fdata:
        print(f"  missing_lines count: {len(fdata['missing_lines'])}")
else:
    print(f"\n{target} NOT FOUND in coverage.json")
    for fp in files:
        if "encoders" in fp:
            print(f"  but found similar: {fp}")

# Check what format source file paths use
print("\n=== File path format analysis ===")
# Distinct unique prefixes
prefixes = set()
for fp in files:
    parts = fp.split("/")
    if len(parts) > 1:
        prefixes.add(parts[0])
print(f"Top-level path prefixes: {sorted(prefixes)}")

# Specifically check the graph's file_path format vs coverage's
print("\n=== Graph Function file_path for fastapi.encoders.jsonable_encoder ===")
print("  (from Q6 diagnostic: fp=fastapi/encoders.py)")
print("  Coverage uses keys like the above file paths")

# Also check the full coverage.json for context names
print("\n=== Sample context names from first file with contexts ===")
for fp, fdata in d["files"].items():
    if "contexts" in fdata and fdata["contexts"]:
        for line, ctxs in list(fdata["contexts"].items())[:1]:
            print(f"  File: {fp}, line {line}")
            for c in ctxs[:5]:
                print(f"    ctx: '{c}'")
        break
