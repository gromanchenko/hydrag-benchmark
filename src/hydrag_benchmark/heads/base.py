"""Base types and protocols for multi-head retrieval benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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
    # T-5052 A11/B-05: per-result provenance from heads (like HeadHydrag)
    # whose head_origin is a static benchmark-head label, not the internal
    # head that actually produced this specific result. Other heads leave
    # this empty. Keys used: "hydrag_head_origin", "fast_path_triggered",
    # "crag_skipped".
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for dense embedding models."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


@runtime_checkable
class RetrievalHead(Protocol):
    """Protocol every Head* class satisfies (T-5061: replaces the
    beir_runner.py dispatch's implicit-first-assignment head variable
    typing, which made every later branch's reassignment to a
    different concrete Head* class a mypy "incompatible types in
    assignment" error).

    build_index is required here: HeadA and HeadC index in their own
    constructor and never need it called, but both still implement it
    as a no-op so every head satisfies one shared interface rather than
    the dispatch code needing casts or per-branch type: ignore.
    close() and the HeadD-only load_corpus_metadata() are deliberately
    NOT part of this Protocol -- not every head is closeable, and
    load_corpus_metadata is a HeadD-specific pre-seeded-snapshot
    feature; both are accessed at their call sites via hasattr/
    isinstance narrowing instead of a falsely-universal Protocol
    member.
    """

    @property
    def name(self) -> str: ...

    def build_index(self, chunks: list[Chunk]) -> object: ...

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]: ...
