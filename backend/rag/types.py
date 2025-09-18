from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import uuid


@dataclass(slots=True)
class DocumentChunk:
    """A small piece of text extracted from a source document."""

    text: str
    source: str
    order: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_metadata(self) -> Dict[str, Any]:
        meta = {"source": self.source, "order": self.order}
        meta.update(self.metadata)
        return meta


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: Dict[str, Any]

    def for_client(self) -> Dict[str, Any]:
        data = {
            "text": self.text,
            "score": round(self.score, 4),
        }
        data.update(self.metadata)
        return data
