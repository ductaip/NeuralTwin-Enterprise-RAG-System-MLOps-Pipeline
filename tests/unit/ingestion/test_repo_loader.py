from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codeatlas.ingestion.repo_loader import SKIP_DIRS, load_repo


def write(root: Path, rel: str, content: str = "x = 1\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_module_names_follow_package_layout(tmp_path: Path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/mod.py")
    write(tmp_path, "pkg/sub/__init__.py", "")
    write(tmp_path, "pkg/sub/deep.py")

    loaded = load_repo(str(tmp_path))
    names = {f.module_name for f in loaded.files}
    assert names == {"pkg", "pkg.mod", "pkg.sub", "pkg.sub.deep"}


def test_skip_dirs_are_excluded(tmp_path: Path):
    write(tmp_path, "real.py")
    for skipped in ("__pycache__", "node_modules", ".venv", "build"):
        write(tmp_path, f"{skipped}/junk.py")

    loaded = load_repo(str(tmp_path))
    assert [f.path for f in loaded.files] == ["real.py"]
    assert SKIP_DIRS >= {"__pycache__", "node_modules", ".venv", "build"}


def test_non_python_files_are_ignored(tmp_path: Path):
    write(tmp_path, "keep.py")
    write(tmp_path, "notes.md", "# hi\n")
    write(tmp_path, "data.json", "{}\n")

    loaded = load_repo(str(tmp_path))
    assert [f.path for f in loaded.files] == ["keep.py"]


def test_gitignored_files_are_respected(tmp_path: Path):
    """.gitignore is honoured by git itself via `git ls-files`, not reimplemented."""
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git unavailable")

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "secret.py\ngenerated/\n")
    write(tmp_path, "kept.py")
    write(tmp_path, "secret.py")
    write(tmp_path, "generated/out.py")

    loaded = load_repo(str(tmp_path))
    paths = {f.path for f in loaded.files}
    assert "kept.py" in paths
    assert "secret.py" not in paths
    assert "generated/out.py" not in paths


def test_sha_changes_with_content(tmp_path: Path):
    write(tmp_path, "a.py", "x = 1\n")
    first = load_repo(str(tmp_path)).files[0].sha
    write(tmp_path, "a.py", "x = 2\n")
    second = load_repo(str(tmp_path)).files[0].sha
    assert first != second


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        load_repo(str(tmp_path / "nope"))
