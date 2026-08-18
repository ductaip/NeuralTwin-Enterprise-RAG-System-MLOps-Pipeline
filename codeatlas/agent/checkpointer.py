"""Checkpointer selection — spec §2.3: SqliteSaver for dev, PostgresSaver for prod,
configured through env. Both are context managers (`from_conn_string`), so callers
must use them as such rather than holding a bare instance past the `with` block.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver

from codeatlas.settings import settings


@contextmanager
def get_checkpointer() -> Iterator[BaseCheckpointSaver]:
    if settings.LANGGRAPH_CHECKPOINT_BACKEND == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver

        if not settings.LANGGRAPH_POSTGRES_URI:
            raise ValueError(
                "LANGGRAPH_CHECKPOINT_BACKEND=postgres but LANGGRAPH_POSTGRES_URI is not set."
            )
        with PostgresSaver.from_conn_string(settings.LANGGRAPH_POSTGRES_URI) as saver:
            saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(settings.LANGGRAPH_SQLITE_PATH) as saver:
            yield saver
