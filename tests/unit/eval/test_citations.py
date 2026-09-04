"""Citation verification + `citation_validity_rate` (spec §3.2).

Guards the live-verified failure: the ReAct loop cited
`src/http/request_parser.py:45-78`, a file that exists nowhere in the indexed repo.
"""

from __future__ import annotations

import pytest

from codeatlas.eval.citations import check_citations, known_sources, strip_invalid_citations

EVIDENCE = [
    {
        "tool": "search_symbol",
        "result": {
            "results": [
                {
                    "qualified_name": "pkg.mod.foo",
                    "source": {"file_path": "pkg/mod.py", "start": 10, "end": 40},
                }
            ]
        },
    },
    {
        "tool": "get_callers",
        "result": {
            "callers": [
                {
                    "qualified_name": "pkg.mod.bar",
                    "source": {"file_path": "pkg/other.py", "start": 100, "end": 120},
                }
            ]
        },
    },
]


def test_known_sources_collects_paths_and_ranges_from_any_tool_shape():
    sources = known_sources(EVIDENCE)
    assert sources == {"pkg/mod.py": [(10, 40)], "pkg/other.py": [(100, 120)]}


def test_valid_citation_counts_as_valid():
    check = check_citations("see [pkg/mod.py:12-30]", EVIDENCE)
    assert (check.total, check.valid) == (1, 1)
    assert check.validity_rate == 1.0


def test_fabricated_file_is_invalid():
    check = check_citations("see [src/http/request_parser.py:45-78]", EVIDENCE)
    assert (check.total, check.valid) == (1, 0)
    assert check.invalid_citations == ["src/http/request_parser.py:45-78"]
    assert check.validity_rate == 0.0


def test_real_file_but_never_retrieved_line_range_is_invalid():
    """Subtler fabrication than inventing a filename: right file, invented lines."""
    check = check_citations("see [pkg/mod.py:900-950]", EVIDENCE)
    assert check.valid == 0
    assert check.invalid_citations == ["pkg/mod.py:900-950"]


def test_line_overlap_check_can_be_relaxed():
    check = check_citations("see [pkg/mod.py:900-950]", EVIDENCE, require_line_overlap=False)
    assert check.valid == 1


def test_answer_with_no_citations_is_perfectly_valid():
    """Nothing claimed means nothing fabricated — 1.0, not 0.0."""
    check = check_citations("không tìm thấy trong codebase", EVIDENCE)
    assert (check.total, check.validity_rate) == (0, 1.0)


def test_mixed_citations_give_partial_validity_rate():
    answer = "real [pkg/mod.py:12-30] and fake [made/up.py:1-2]"
    check = check_citations(answer, EVIDENCE)
    assert (check.total, check.valid) == (2, 1)
    assert check.validity_rate == pytest.approx(0.5)


def test_strip_removes_only_the_invalid_citation():
    answer = "real [pkg/mod.py:12-30] and fake [made/up.py:1-2]"
    cleaned, check = strip_invalid_citations(answer, EVIDENCE)
    assert "[pkg/mod.py:12-30]" in cleaned
    assert "[made/up.py:1-2]" not in cleaned.split("[Đã gỡ")[0]
    assert check.validity_rate == pytest.approx(0.5)


def test_strip_notes_what_it_removed_rather_than_silently_rewriting():
    cleaned, _ = strip_invalid_citations("bogus [made/up.py:1-2]", EVIDENCE)
    assert "Đã gỡ" in cleaned
    assert "made/up.py:1-2" in cleaned


def test_strip_leaves_clean_answers_untouched():
    answer = "all good [pkg/other.py:100-120]"
    cleaned, check = strip_invalid_citations(answer, EVIDENCE)
    assert cleaned == answer
    assert check.validity_rate == 1.0


def test_bracket_variant_used_by_the_model_is_also_caught():
    """The live failure used 【...】, not [...] — both must be checked."""
    check = check_citations("see 【5†src/http/request_parser.py:45-78】", EVIDENCE)
    assert check.total == 1
    assert check.valid == 0
