"""Base types and protocols for multi-head retrieval benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Chunk:
    """A single text chunk from the corpus.

    Attributes:
        chunk_id: Content-addressed ID (SHA-256 hex prefix via ``content_address()``).
            Callers may also use sequential or external IDs.
        text: Raw chunk text content.
        source: Origin identifier (e.g. file path, URL, or corpus document ID).
        symbols: Optional code symbols extracted from the chunk (function names,
            class names, etc.). Empty list for prose content.
    """

    chunk_id: str
    text: str
    source: str
    symbols: list[str] = field(default_factory=list)

    @staticmethod
    def content_address(text: str) -> str:
        """SHA-256 content-addressed ID (first 16 hex chars)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScoredChunk:
    """A chunk with a retrieval score and provenance."""

    chunk: Chunk
    score: float
    head_origin: str


@runtime_checkable
class Embedder(Protocol):
    """Protocol for dense embedding models."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


@runtime_checkable
class RetrievalHead(Protocol):
    """Protocol for a retrieval head."""

    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]: ...
