"""Split code identifiers into sub-tokens for BM25, keeping the original token too.

`getUserById` -> `getuserbyid get user by id` (lowercased for case-insensitive matching).
Keeping the original alongside the split form matters: a query for the exact identifier
should still get an exact-token match, not just a bag of sub-words.
"""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_WORD_RE = re.compile(r"[a-zA-Z]+|[0-9]+")

# No English stopword list: "to", "in", "for", "with" are real signal in identifiers like
# `to_json`, `for_each`, `with_context`, `in_place` — filtering them as prose filler would
# break exactly the compound names BM25 needs to match on. Frequency weighting (IDF) is
# BM25's job, not the tokenizer's.


def split_identifier(identifier: str) -> list[str]:
    """Split one identifier into lowercase sub-tokens: camelCase, snake_case, and digits."""
    parts = _NON_ALNUM.split(identifier)
    sub_tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        camel_split = _CAMEL_BOUNDARY.sub(" ", part)
        sub_tokens.extend(m.group(0).lower() for m in _WORD_RE.finditer(camel_split))
    return sub_tokens


def tokenize(text: str) -> list[str]:
    """Tokenize a chunk of text/code for BM25: split on non-identifier characters, then
    split each identifier into sub-tokens, keeping the whole identifier as an extra token.
    """
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", text):
        raw_lower = raw.lower()
        sub_tokens = split_identifier(raw)
        if len(sub_tokens) > 1:
            tokens.append(raw_lower)
            tokens.extend(sub_tokens)
        elif sub_tokens:
            tokens.append(sub_tokens[0])
        else:
            tokens.append(raw_lower)

    return [t for t in tokens if len(t) > 1]
