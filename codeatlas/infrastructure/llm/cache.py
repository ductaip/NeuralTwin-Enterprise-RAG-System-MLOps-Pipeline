"""Disk cache shared by every LLM provider.

Key = (provider, model, prompt_hash, temperature, seed) per spec §2.6 — re-running an
ablation or a demo query costs nothing once every prompt has been seen once.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import diskcache

from codeatlas.settings import settings

_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        path = Path(settings.LLM_CACHE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(path))
    return _cache


def cache_key(provider: str, model: str, prompt: str, temperature: float, seed: Any) -> str:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{provider}:{model}:{prompt_hash}:{temperature}:{seed}"
