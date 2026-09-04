"""Check coverage_full.json for fastapi/encoders.py contexts (streaming to avoid OOM)."""
import json
import sys

cov_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fastapi_codeatlas/coverage_full.json"

print(f"Loading {cov_path}...")
with open(cov_path) as f:
    d = json.load(f)

files = list(d.get("files", {}).keys())
print(f"Total source files: {len(files)}")

target = "fastapi/encoders.py"
if target in d["files"]:
    fdata = d["files"][target]
    print(f"\n{target} FOUND")
    print(f"  has 'contexts': {'contexts' in fdata}")
    if "contexts" in fdata:
        ctx = fdata["contexts"]
        print(f"  num lines with contexts: {len(ctx)}")
        
        non_empty_lines = 0
        non_empty_ctxs = set()
        for line, contexts in ctx.items():
            real_ctxs = [c for c in contexts if c]  # filter empty strings
            if real_ctxs:
                non_empty_lines += 1
                for c in real_ctxs:
                    non_empty_ctxs.add(c)
        
        print(f"  lines with NON-EMPTY contexts: {non_empty_lines}")
        print(f"  unique non-empty context names: {len(non_empty_ctxs)}")
        
        # Show samples
        if non_empty_ctxs:
            print("\n  Sample non-empty contexts:")
            for c in sorted(non_empty_ctxs)[:15]:
                print(f"    '{c}'")
        
        # Show lines in jsonable_encoder range (129-366)
        print("\n  Lines 129-366 (jsonable_encoder body):")
        encoder_ctxs = set()
        for line_s, contexts in ctx.items():
            line = int(line_s)
            if 129 <= line <= 366:
                real = [c for c in contexts if c]
                if real:
                    encoder_ctxs.update(real)
                    if len(encoder_ctxs) <= 5:
                        print(f"    line {line}: {real[:3]}{'...' if len(real)>3 else ''}")
        print(f"    Total unique test contexts hitting jsonable_encoder: {len(encoder_ctxs)}")
        if encoder_ctxs:
            print("    Samples:")
            for c in sorted(encoder_ctxs)[:10]:
                print(f"      '{c}'")
    else:
        print("  No 'contexts' key at all")
        
    if "executed_lines" in fdata:
        el = fdata["executed_lines"]
        enc_lines = [l for l in el if 129 <= l <= 366]
        print(f"\n  executed_lines in jsonable_encoder range: {len(enc_lines)} of {len(el)} total")
else:
    print(f"\n{target} NOT FOUND")
    for fp in files:
        if "encoder" in fp.lower():
            print(f"  similar: {fp}")

# Also sample a file that HAS non-empty contexts to see what format
print("\n=== File with most non-empty contexts (for format reference) ===")
best_file = None
best_count = 0
for fp, fdata in d["files"].items():
    if "contexts" in fdata:
        c = sum(1 for line, ctxs in fdata["contexts"].items() if any(c for c in ctxs))
        if c > best_count:
            best_count = c
            best_file = fp

if best_file:
    print(f"  File: {best_file} ({best_count} lines with non-empty contexts)")
    fdata = d["files"][best_file]
    # show a few
    shown = 0
    for line, ctxs in sorted(fdata["contexts"].items(), key=lambda x: int(x[0])):
        real = [c for c in ctxs if c]
        if real and shown < 3:
            print(f"    line {line}: {real[:3]}{'...' if len(real)>3 else ''}")
            shown += 1
