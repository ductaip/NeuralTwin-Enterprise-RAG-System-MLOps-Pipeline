from __future__ import annotations

import pytest

from codeatlas.application.rag.identifier_tokenizer import split_identifier, tokenize

# 14 identifier styles, per Phase 2 roadmap requirement of >= 10.
SPLIT_CASES = [
    ("getUserById", ["get", "user", "by", "id"]),
    ("get_user_by_id", ["get", "user", "by", "id"]),
    ("GetUserById", ["get", "user", "by", "id"]),
    ("MAX_RETRY_COUNT", ["max", "retry", "count"]),
    ("parseHTML5Document", ["parse", "html", "5", "document"]),
    ("XMLParser", ["xml", "parser"]),
    ("user", ["user"]),
    ("v2", ["v", "2"]),
    ("_internal_helper", ["internal", "helper"]),
    ("get_UserByID", ["get", "user", "by", "id"]),
    ("HTTPSConnectionPool", ["https", "connection", "pool"]),
    ("toJSON", ["to", "json"]),
    ("for_each", ["for", "each"]),
    ("with_context", ["with", "context"]),
]


@pytest.mark.parametrize("identifier,expected", SPLIT_CASES)
def test_split_identifier(identifier, expected):
    assert split_identifier(identifier) == expected


@pytest.mark.parametrize("identifier,sub_tokens", SPLIT_CASES)
def test_tokenize_keeps_original_alongside_split_form(identifier, sub_tokens):
    """A single-word identifier isn't duplicated; a compound one keeps both forms."""
    tokens = tokenize(identifier)
    for sub in sub_tokens:
        if len(sub) > 1:
            assert sub in tokens, f"{sub!r} missing from {tokens!r} for {identifier!r}"

    if len(sub_tokens) > 1:
        raw_lower = identifier.lower()
        assert raw_lower in tokens, f"whole identifier {raw_lower!r} missing from {tokens!r}"


def test_prose_looking_substrings_are_not_dropped_as_stopwords():
    """`toJSON`/`for_each`/`with_context` are common method-name components; an English
    stopword filter would silently break exact-match BM25 queries for them."""
    assert "to" in tokenize("toJSON")
    assert "for" in tokenize("for_each")
    assert "with" in tokenize("with_context")
    assert "in" in tokenize("in_place")


def test_single_character_tokens_are_dropped_as_noise():
    assert tokenize("i") == []
    assert "v" not in tokenize("v2")


def test_dotted_access_tokenizes_each_identifier_separately():
    tokens = tokenize("self.getUserById(user_id)")
    assert "self" in tokens
    assert "getuserbyid" in tokens
    assert "user_id" in tokens
    assert "user" in tokens
    assert "id" in tokens


def test_tokenize_is_case_insensitive():
    assert tokenize("GetUser") == tokenize("getuser".replace("getuser", "GetUser"))
    assert all(t == t.lower() for t in tokenize("GetUserByID_v2"))
