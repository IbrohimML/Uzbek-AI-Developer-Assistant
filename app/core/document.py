from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.content, str):
            raise TypeError("Document content must be a string")

        if not self.content.strip():
            raise ValueError("Document content cannot be empty")