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

    def _fts_search_with_ids(self, query: str, n_results: int) -> list[tuple[str, str]]:
        """FTS5 search returning (chunk_id, raw_content) pairs with BM25 ranking."""
        safe_query = self._store._escape_fts_query(query)
        if not safe_query:
            return []
        try:
            rows = self._store._conn.execute(
                """SELECT c.chunk_id, c.raw_content
                   FROM chunks_fts f
                   JOIN chunks c ON c.rowid = f.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY bm25(chunks_fts, 1.0, 1.0, 0.8, 1.4)
                   LIMIT ?""",
                (safe_query, n_results),
            ).fetchall()
            return [(row["chunk_id"], row["raw_content"]) for row in rows]
        except Exception:
            logger.debug("FTS query failed: %s", safe_query, exc_info=True)
            return []

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """FTS5 BM25 retrieval."""
        hits = self._fts_search_with_ids(query, n_results)
        results: list[ScoredChunk] = []
        for rank, (chunk_id, _text) in enumerate(hits):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
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
