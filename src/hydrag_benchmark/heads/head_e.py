"""Head E — Lexical retrieval + LLM enrichment.

Same lexical engine as Head D, but with OllamaKeywordExtractor enrichment
at index time. Each chunk gets LLM-generated summary + keywords stored
in FTS5 columns with weighted BM25 ranking (SQLite backend only):
  - raw_content: weight 1.0
  - summary: weight 0.8
  - keywords: weight 1.4

When using a non-SQLite backend (e.g. SurrealDB), enrichment is skipped
and the head behaves identically to Head D for that backend.

Measures the lift from T-743 enrichment over raw FTS5 (Head D).
Requires a running Ollama instance. Index time is slower (LLM calls),
but query time remains sub-millisecond.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydrag import IndexedChunk, SQLiteFTSStore
from hydrag.enrichment import OllamaKeywordExtractor

from .base import Chunk
from .head_d import HeadD

logger = logging.getLogger("hydrag_benchmark.heads.head_e")


class HeadE(HeadD):
    """Lexical retrieval + LLM enrichment head.

    Subclasses HeadD — inherits query and ranking logic.
    Index-time: chunks enriched with summary + keywords via Ollama
    (SQLite backend only), then inserted into the store.
    Query-time: BM25 ranking via ``adapter.keyword_search()``.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        model: str = "qwen3:4b",
        ollama_host: str = "http://localhost:11434",
        *,
        adapter: Any = None,
    ) -> None:
        super().__init__(db_path, adapter=adapter)
        self._extractor = OllamaKeywordExtractor(
            model=model, host=ollama_host,
        )

    @property
    def name(self) -> str:
        return "head_e"

    def build_index(self, chunks: list[Chunk]) -> None:
        """Index chunks with LLM enrichment (SQLite) or raw content (other backends)."""
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
        if isinstance(self._store, SQLiteFTSStore):
            count = self._store.index_documents(
                indexed_chunks,
                extractor=self._extractor,
                model_id=self._extractor.model,
                prompt_hash=self._extractor.prompt_hash,
            )
            logger.info("Head E indexed %d chunks (FTS5 + enrichment)", count)
        else:
            logger.info(
                "Enrichment not supported with %s backend; indexing raw content",
                type(self._store).__name__,
            )
            count = self._store.index_documents(indexed_chunks)
            logger.info("Head E indexed %d chunks (raw)", count)

    # retrieve() is inherited from HeadD — uses self.name for head_origin

    def __enter__(self) -> "HeadE":
        return self
