"""Head D — SQLite FTS5 lexical retrieval. Zero GPU, zero network.

Wraps hydrag-core's SQLiteFTSStore as a benchmark RetrievalHead.
BM25 ranking over raw content only (no enrichment). Sub-millisecond
query latency. Baseline for measuring enrichment lift (Head E).
"""

from __future__ import annotations

import logging
from pathlib import Path

from hydrag import IndexedChunk, SQLiteFTSStore

from .base import Chunk, ScoredChunk

logger = logging.getLogger("hydrag_benchmark.heads.head_d")


class HeadD:
    """SQLite FTS5 lexical retrieval head. No GPU, no network.

    Index-time: chunks are inserted into SQLite FTS5 (raw content only).
    Query-time: BM25 ranking over FTS5 index, sub-millisecond latency.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._store = SQLiteFTSStore(db_path)
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
                head_origin="head_d",
            ))
        return results

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "HeadD":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
