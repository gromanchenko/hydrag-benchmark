"""Head E — SQLite FTS5 + LLM enrichment retrieval.

Same FTS5 engine as Head D, but with OllamaKeywordExtractor enrichment
at index time. Each chunk gets LLM-generated summary + keywords stored
in FTS5 columns with weighted BM25 ranking:
  - raw_content: weight 1.0
  - summary: weight 0.8
  - keywords: weight 1.4

Measures the lift from T-743 enrichment over raw FTS5 (Head D).
Requires a running Ollama instance. Index time is slower (LLM calls),
but query time remains sub-millisecond.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hydrag import IndexedChunk
from hydrag.enrichment import OllamaKeywordExtractor

from .base import Chunk, ScoredChunk
from .head_d import HeadD
from .head_ids import HEAD_ID_ALIASES

logger = logging.getLogger("hydrag_benchmark.heads.head_e")


class HeadE(HeadD):
    """SQLite FTS5 + LLM enrichment retrieval head.

    Subclasses HeadD — inherits FTS5 query and ranking logic.
    Index-time: chunks enriched with summary + keywords via Ollama,
    then inserted into SQLite FTS5 with column weights.
    Query-time: BM25 ranking, sub-millisecond latency.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        model: str = "qwen3:4b",
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        super().__init__(db_path)
        self._extractor = OllamaKeywordExtractor(
            model=model, host=ollama_host,
        )

    @property
    def name(self) -> str:
        return HEAD_ID_ALIASES["head_e"]

    def build_index(self, chunks: list[Chunk]) -> None:
        """Index chunks into FTS5 with LLM enrichment."""
        indexed_chunks = []
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            # B-01: _fts_search_chunk_ids (inherited from HeadD) maps FTS5
            # results back to chunk_id via self._text_to_id, keyed on
            # raw_content (== chunk.text -- see SQLiteFTSStore._fts_search,
            # which always SELECTs raw_content regardless of enrichment).
            # This head never populated it, so every real query returned
            # zero results.
            self._text_to_id[chunk.text] = chunk.chunk_id
            indexed_chunks.append(IndexedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title="",
                raw_content=chunk.text,
            ))
        count = self._store.index_documents(
            indexed_chunks,
            extractor=self._extractor,
            model_id=self._extractor.model,
            prompt_hash=self._extractor.prompt_hash,
        )
        logger.info("Head E indexed %d chunks (FTS5 + enrichment)", count)

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """FTS5 BM25 retrieval (with enriched columns)."""
        chunk_ids = self._fts_search_chunk_ids(query, n_results)
        results: list[ScoredChunk] = []
        for rank, chunk_id in enumerate(chunk_ids):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            results.append(ScoredChunk(
                chunk=chunk,
                score=1.0 / (rank + 1),
                head_origin=self.name,
            ))
        return results

    def __enter__(self) -> "HeadE":
        return self
