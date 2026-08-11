from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RawContentEvent(BaseEvent):
    """Event representing raw content collected from a source."""
    source_url: str
    text: str
    platform: str
    author_id: Optional[str] = None


class ProcessedDocumentEvent(BaseEvent):
    """Event representing a cleaned and processed document ready for embedding."""
    original_event_id: UUID
    content: str
    chunk_index: int = 0
    total_chunks: int = 1
