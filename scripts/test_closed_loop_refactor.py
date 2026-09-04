"""Demonstrate full closed-loop refactor execution trace: PASS -> FAIL -> REPAIR -> PASS."""
import time
from pathlib import Path
import subprocess

from codeatlas.agent.refactor_graph import add_refactor_nodes
from codeatlas.agent.tools import AgentTools
from codeatlas.agent.trace import AgentTracer
from langgraph.graph import StateGraph, START, END

sandbox_dir = Path("/tmp/fastapi_codeatlas")
encoder_file = sandbox_dir / "fastapi" / "encoders.py"

# Make sure git working directory is clean in sandbox
subprocess.run(["git", "checkout", "fastapi/encoders.py"], cwd=str(sandbox_dir), capture_output=True)

print("=== 1. Testing clean state (Pass) ===")
pytest_bin = str(sandbox_dir / ".venv" / "bin" / "pytest")
cmd = [pytest_bin, "tests/test_jsonable_encoder.py"]
res1 = subprocess.run(cmd, cwd=str(sandbox_dir), capture_output=True, text=True)
print(f"Return code: {res1.returncode}")
print(f"Summary line: {res1.stdout.strip().splitlines()[-1] if res1.stdout else ''}")

print("\n=== 2. Injecting breaking change into sandbox fastapi/encoders.py ===")
original_code = encoder_file.read_text()
# Break jsonable_encoder
broken_code = original_code.replace(
    "def jsonable_encoder(",
    "def jsonable_encoder(*args, **kwargs):\n    raise RuntimeError('Refactor test injection error')\n\ndef _old_jsonable_encoder("
)
encoder_file.write_text(broken_code)

res2 = subprocess.run(cmd, cwd=str(sandbox_dir), capture_output=True, text=True)
print(f"Return code after breaking patch: {res2.returncode}")
print(f"Failure line sample: {[l for l in res2.stdout.splitlines() if 'FAILED' in l or 'RuntimeError' in l][:3]}")

# Test check_tests logic
output = res2.stdout + "\n" + res2.stderr
has_failed = "FAILED" in output or "failed" in output.lower() or "error" in output.lower()
print(f"check_tests detection: has_failed={has_failed} -> route: {'repair' if has_failed else 'human_approval'}")

print("\n=== 3. Restoring original code (Simulating Repair Patch) ===")
encoder_file.write_text(original_code)
res3 = subprocess.run(cmd, cwd=str(sandbox_dir), capture_output=True, text=True)
print(f"Return code after repair patch: {res3.returncode}")
print(f"Summary line: {res3.stdout.strip().splitlines()[-1] if res3.stdout else ''}")

has_failed_after = "FAILED" in res3.stdout or "failed" in res3.stdout.lower()
print(f"check_tests detection after repair: has_failed={has_failed_after} -> route: {'human_approval' if not has_failed_after else 'repair'}")
