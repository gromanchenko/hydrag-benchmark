"""Head D — Lexical (BM25) retrieval. Zero GPU, zero network.

Wraps a VectorStoreAdapter backend (default: SQLiteFTSStore) as a
benchmark RetrievalHead. BM25 ranking over raw content only (no
enrichment). Sub-millisecond query latency with the SQLite backend.
Baseline for measuring enrichment lift (Head E).

Pass ``adapter`` to swap the storage backend (e.g. SurrealDBAdapter).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydrag import IndexedChunk, SQLiteFTSStore

from .base import Chunk, ScoredChunk

logger = logging.getLogger("hydrag_benchmark.heads.head_d")


class HeadD:
    """Lexical retrieval head — pluggable storage backend.

    Index-time: chunks are inserted via ``adapter.index_documents()``.
    Query-time: BM25 ranking via ``adapter.keyword_search()``.

    Args:
        db_path: SQLite database path (used only when *adapter* is None).
        adapter: Pre-configured VectorStoreAdapter. When provided,
            *db_path* is ignored and the adapter is used directly.
    """

    def __init__(self, db_path: str | Path = ":memory:", *, adapter: Any = None) -> None:
        self._store = adapter if adapter is not None else SQLiteFTSStore(db_path)
        self._chunks: dict[str, Chunk] = {}
        self._text_to_id: dict[str, str] = {}  # content hash → chunk_id

    @property
    def name(self) -> str:
        return "head_d"

    def build_index(self, chunks: list[Chunk]) -> None:
        """Index chunks into FTS5. No enrichment."""
        indexed_chunks = []
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._text_to_id[chunk.text] = chunk.chunk_id
            indexed_chunks.append(IndexedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title="",
                raw_content=chunk.text,
            ))
        count = self._store.index_documents(indexed_chunks)
        logger.info("Head D indexed %d chunks (FTS5 raw)", count)

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """FTS5 BM25 retrieval via standard adapter interface."""
        texts = self._store.keyword_search(query, n_results)
        results: list[ScoredChunk] = []
        for rank, text in enumerate(texts):
            chunk_id = self._text_to_id.get(text)
            if not chunk_id:
                continue
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue
            results.append(ScoredChunk(
                chunk=chunk,
                score=1.0 / (rank + 1),
                head_origin=self.name,
            ))
        return results

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "HeadD":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
