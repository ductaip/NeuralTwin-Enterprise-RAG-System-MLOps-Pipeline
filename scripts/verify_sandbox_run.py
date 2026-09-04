"""Verify real sandbox pytest run and capture raw stdout/stderr."""
import os
import subprocess
from pathlib import Path

sandbox_dir = Path("/tmp/fastapi_codeatlas")
pytest_bin = str(sandbox_dir / ".venv" / "bin" / "pytest")

print(f"Sandbox dir exists: {sandbox_dir.exists()}")
print(f"Pytest bin exists: {os.path.exists(pytest_bin)} ({pytest_bin})")

# Run pytest on tests/test_jsonable_encoder.py in sandbox
cmd = [pytest_bin, "tests/test_jsonable_encoder.py"]
print(f"\nRunning command: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=str(sandbox_dir), capture_output=True, text=True)

print(f"Return code: {result.returncode}")
print("\n--- RAW STDOUT ---")
print(result.stdout)
print("\n--- RAW STDERR ---")
print(result.stderr)
