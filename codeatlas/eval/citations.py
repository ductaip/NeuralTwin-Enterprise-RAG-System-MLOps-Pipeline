"""Citation verification — the mechanical guard against fabricated `[file.py:12-30]`.

Live-verified failure this exists for: the custom ReAct loop answered
"Request body parsing is implemented in `parse_body` inside
`src/http/request_parser.py` 【src/http/request_parser.py:45-78】" — a file that
exists nowhere in the indexed repo. CLAUDE.md's hard rule is that a fabricated citation
must never be presented as legitimate, so an unverifiable one is **removed** from the
answer, not merely flagged: a reader who sees a warning still tends to trust the rest of
the sentence around it.

`citation_validity_rate` is the Phase 5 metric built on the same check (spec §3.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"[\[【]\s*(?:\d+†)?([\w./\\-]+\.\w+):(\d+)-(\d+)\s*[\]】]")


@dataclass
class CitationCheck:
    total: int = 0
    valid: int = 0
    invalid_citations: list[str] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        """% of citations pointing at a file/line range that actually appeared in
        evidence. 1.0 when an answer makes no citations at all — nothing was claimed,
        so nothing was fabricated."""
        return self.valid / self.total if self.total else 1.0


def known_sources(evidence: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """Map `file_path -> [(start, end), ...]` for every source that appeared in a real
    tool result, whatever shape that tool returns (`source`, `results[].source`,
    `callers[].source`, `impacted_symbols[].source`, ...)."""
    sources: dict[str, list[tuple[int, int]]] = {}

    def walk(node) -> None:
        if isinstance(node, dict):
            fp = node.get("file_path")
            if isinstance(fp, str):
                start, end = node.get("start"), node.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    start, end = node.get("start_line"), node.get("end_line")
                if isinstance(start, int) and isinstance(end, int):
                    sources.setdefault(fp, []).append((start, end))
                else:
                    sources.setdefault(fp, [])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for item in evidence:
        walk(item.get("result", item))
    return sources


def check_citations(answer: str, evidence: list[dict], require_line_overlap: bool = True) -> CitationCheck:
    """Validate every citation in `answer` against `evidence`.

    `require_line_overlap=True` also rejects a citation naming a real file but a line
    range never actually retrieved — a subtler fabrication than inventing a filename.
    """
    sources = known_sources(evidence)
    result = CitationCheck()

    for file_path, start_s, end_s in CITATION_RE.findall(answer):
        result.total += 1
        raw = f"{file_path}:{start_s}-{end_s}"
        ranges = sources.get(file_path)
        if ranges is None:
            result.invalid_citations.append(raw)
            continue
        if not require_line_overlap or not ranges:
            result.valid += 1
            continue
        start, end = int(start_s), int(end_s)
        if any(start <= r_end and end >= r_start for r_start, r_end in ranges):
            result.valid += 1
        else:
            result.invalid_citations.append(raw)

    return result


def strip_invalid_citations(answer: str, evidence: list[dict]) -> tuple[str, CitationCheck]:
    """Remove every unverifiable citation from `answer` and note what was removed.

    Removal rather than flagging is deliberate — see module docstring. The trailing note
    keeps the removal visible instead of silently rewriting the model's output.
    """
    check = check_citations(answer, evidence)
    if not check.invalid_citations:
        return answer, check

    invalid = set(check.invalid_citations)

    def replace(match: re.Match) -> str:
        raw = f"{match.group(1)}:{match.group(2)}-{match.group(3)}"
        return "" if raw in invalid else match.group(0)

    cleaned = CITATION_RE.sub(replace, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    removed = ", ".join(sorted(invalid))
    return (
        f"{cleaned}\n\n[Đã gỡ {len(invalid)} trích dẫn không khớp bằng chứng thu thập "
        f"được: {removed}. Nội dung dựa trên các trích dẫn này không kiểm chứng được.]",
        check,
    )
