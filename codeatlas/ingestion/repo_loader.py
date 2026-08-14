"""Turn a git URL or a local path into the list of source files to index."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from codeatlas.ingestion.models import SourceFile

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "venv",
        ".venv",
        "env",
        ".env",
        "build",
        "dist",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "site-packages",
    }
)


@dataclass
class LoadedRepo:
    repo_id: str
    root: Path
    commit_sha: str | None
    files: list[SourceFile] = field(default_factory=list)
    cleanup_dir: Path | None = None
    """Temp directory to remove when done, if we cloned."""

    @property
    def indexed_modules(self) -> set[str]:
        return {f.module_name for f in self.files}


def _content_sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _current_commit(root: Path) -> str | None:
    try:
        return _git(root, "rev-parse", "HEAD").strip()
    except RuntimeError:
        return None


def _list_python_files_git(root: Path) -> list[Path]:
    """Use `git ls-files` so .gitignore is honoured by git itself rather than reimplemented.

    Includes tracked files plus untracked-but-not-ignored ones, matching what a developer
    would consider "the source in this working tree".
    """
    out = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "*.py")
    return [root / p for p in out.split("\0") if p]


def _list_python_files_walk(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for filename in filenames:
            if filename.endswith(".py"):
                found.append(Path(dirpath) / filename)
    return found


def _module_name_for(path: Path, root: Path) -> str:
    """Derive the dotted module path, walking up while `__init__.py` files exist.

    A file outside any package becomes a top-level module named after the file, which is
    what Python itself would do.
    """
    rel = path.relative_to(root)
    parts = list(rel.parts)

    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]

    # Trim leading directories that are not packages (no __init__.py), e.g. a `src/` layout.
    while parts:
        candidate_dir = root.joinpath(*parts[:1])
        if candidate_dir.is_dir() and not (candidate_dir / "__init__.py").exists():
            parts = parts[1:]
        else:
            break

    return ".".join(parts) if parts else rel.stem


def clone_repo(url: str, commit: str | None = None, dest: Path | None = None) -> tuple[Path, Path | None]:
    """Clone `url`. Returns (root, cleanup_dir). `cleanup_dir` is None if `dest` was given."""
    cleanup_dir: Path | None = None
    if dest is None:
        cleanup_dir = Path(tempfile.mkdtemp(prefix="codeatlas-"))
        dest = cleanup_dir / "repo"

    logger.info(f"Cloning {url} -> {dest}")
    clone_args = ["git", "clone", "--quiet"]
    if commit is None:
        clone_args += ["--depth", "1"]
    clone_args += [url, str(dest)]

    result = subprocess.run(clone_args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

    if commit:
        _git(dest, "checkout", "--quiet", commit)

    return dest, cleanup_dir


def load_repo(
    source: str,
    commit: str | None = None,
    repo_id: str | None = None,
    dest: Path | None = None,
) -> LoadedRepo:
    """Load a repository from a git URL or a local path."""
    cleanup_dir: Path | None = None

    if source.startswith(("http://", "https://", "git@", "ssh://")):
        root, cleanup_dir = clone_repo(source, commit=commit, dest=dest)
        default_id = source.rstrip("/").split("/")[-1].removesuffix(".git")
    else:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")
        if commit:
            _git(root, "checkout", "--quiet", commit)
        default_id = root.name

    if _is_git_repo(root):
        try:
            paths = _list_python_files_git(root)
        except RuntimeError as e:
            logger.warning(f"git ls-files failed ({e}); falling back to filesystem walk.")
            paths = _list_python_files_walk(root)
    else:
        paths = _list_python_files_walk(root)

    # `git ls-files` does not know about our skip list, so apply it either way.
    paths = [p for p in paths if not (set(p.relative_to(root).parts) & SKIP_DIRS)]

    files: list[SourceFile] = []
    for path in sorted(paths):
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning(f"Skipping unreadable file {path}: {e}")
            continue

        files.append(
            SourceFile(
                path=path.relative_to(root).as_posix(),
                module_name=_module_name_for(path, root),
                content=content,
                sha=_content_sha(content),
                loc=content.count("\n") + 1,
            )
        )

    logger.info(f"Loaded {len(files)} Python files from {root}")

    return LoadedRepo(
        repo_id=repo_id or default_id,
        root=root,
        commit_sha=_current_commit(root),
        files=files,
        cleanup_dir=cleanup_dir,
    )
