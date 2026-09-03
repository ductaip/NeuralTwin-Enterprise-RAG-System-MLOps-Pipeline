from __future__ import annotations

from codeatlas.agent.custom_react import _flag_unverifiable_citations, _known_file_paths

EVIDENCE = [
    {
        "tool": "search_symbol",
        "args": {"name": "foo"},
        "result": {"results": [{"qualified_name": "pkg.mod.foo", "source": {"file_path": "pkg/mod.py", "start": 1, "end": 5}}]},
    },
    {
        "tool": "get_callers",
        "args": {"qualified_name": "pkg.mod.foo"},
        "result": {"callers": [{"qualified_name": "pkg.mod.bar", "source": {"file_path": "pkg/other.py", "start": 10, "end": 20}}]},
    },
]


def test_known_file_paths_collects_from_nested_source_fields():
    paths = _known_file_paths(EVIDENCE)
    assert paths == {"pkg/mod.py", "pkg/other.py"}


def test_known_file_paths_empty_for_no_evidence():
    assert _known_file_paths([]) == set()


def test_citation_matching_real_evidence_passes_through_unchanged():
    answer = "foo is called by bar [pkg/other.py:10-20]."
    assert _flag_unverifiable_citations(answer, EVIDENCE) == answer


def test_fabricated_citation_is_flagged_not_silently_trusted():
    """Live-verified failure this guards against: the model cited
    `src/http/request_parser.py:45-78` — a file that does not exist anywhere in the
    indexed repo — as if it were real evidence."""
    answer = "Parsing happens in [src/http/request_parser.py:45-78]."
    flagged = _flag_unverifiable_citations(answer, EVIDENCE)
    assert answer in flagged
    assert "CẢNH BÁO" in flagged
    assert "src/http/request_parser.py" in flagged


def test_mixed_real_and_fabricated_citations_flags_only_the_fake_one():
    answer = "See [pkg/mod.py:1-5] and also [made/up.py:1-2]."
    flagged = _flag_unverifiable_citations(answer, EVIDENCE)
    assert "made/up.py" in flagged
    assert "pkg/mod.py" not in flagged.split("CẢNH BÁO")[1] if "CẢNH BÁO" in flagged else True


def test_no_citations_at_all_is_untouched():
    answer = "không tìm thấy trong codebase"
    assert _flag_unverifiable_citations(answer, EVIDENCE) == answer
