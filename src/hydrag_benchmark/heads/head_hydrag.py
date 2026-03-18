"""Head HydRAG — Full multi-headed HydRAG pipeline as a BEIR benchmark head.

Wraps hydrag-core's ``hydrag_search()`` orchestrator (Head 0–3b) behind the
benchmark ``RetrievalHead`` protocol. This lets the BEIR runner compare the
full HydRAG pipeline against individual heads (D, E) on the same datasets.

The adapter uses SQLiteFTSStore as the VectorStoreAdapter backend, matching
Head D/E's indexing. CRAG supervisor and semantic fallback are enabled by
default (web fallback is disabled — offline benchmarking).

Since ``hydrag_search()`` returns ``list[RetrievalResult]`` (text strings),
we maintain a reverse index (text → chunk_id) built at index time to map
results back to BEIR doc IDs for metric computation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from hydrag import HydRAGConfig, IndexedChunk, SQLiteFTSStore, hydrag_search
from hydrag.protocols import LLMProvider

from .base import Chunk, ScoredChunk

logger = logging.getLogger("hydrag_benchmark.heads.head_hydrag")


def _text_hash(text: str) -> str:
    """Stable hash for reverse-index key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HeadHydrag:
    """Full HydRAG multi-headed retrieval benchmark head.

    Index-time: chunks inserted into SQLiteFTSStore (same as Head D).
    Query-time: ``hydrag_search()`` runs the full 5-head pipeline:
      Head 0 — BM25 fast path
      Head 1 — hybrid retrieval
      Head 2 — CRAG supervisor
      Head 3a — semantic fallback (on INSUFFICIENT)
      Head 3b — web fallback (DISABLED for benchmarks)

    Config knobs:
      - ``enable_crag``: run CRAG supervisor (default: True)
      - ``profile``: "prose" or "code" (default: "prose")
      - ``ollama_host`` / ``ollama_model``: for CRAG LLM calls
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        enable_crag: bool = True,
        profile: str = "prose",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "qwen3:4b",
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self._store = SQLiteFTSStore(db_path)
        self._chunks: dict[str, Chunk] = {}
        self._text_to_chunk_id: dict[str, str] = {}  # text_hash → chunk_id
        self._llm = llm

        self._config = HydRAGConfig(
            profile=profile,
            crag_model=ollama_model,
            ollama_host=ollama_host,
            enable_head_0=True,
            enable_head_1=True,
            enable_head_2_crag=enable_crag,
            enable_head_3a_semantic=True,
            enable_head_3b_web=False,  # offline benchmarks
            enable_web_fallback=False,
        )

    @property
    def name(self) -> str:
        return "head_hydrag"

    def build_index(self, chunks: list[Chunk]) -> None:
        """Index chunks into SQLiteFTSStore and build reverse text→chunk_id map."""
        indexed_chunks: list[IndexedChunk] = []
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._text_to_chunk_id[_text_hash(chunk.text)] = chunk.chunk_id
            indexed_chunks.append(IndexedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title="",
                raw_content=chunk.text,
            ))
        count = self._store.index_documents(indexed_chunks)
        logger.info("HeadHydrag indexed %d chunks (SQLiteFTS)", count)

    def _resolve_chunk_id(self, text: str) -> str | None:
        """Map a retrieved text string back to its chunk_id via reverse index."""
        key = _text_hash(text)
        chunk_id = self._text_to_chunk_id.get(key)
        if chunk_id:
            return chunk_id
        # Fallback: prefix match (hydrag_search may truncate text)
        for stored_key, cid in self._text_to_chunk_id.items():
            stored_chunk = self._chunks.get(cid)
            if stored_chunk and (
                stored_chunk.text.startswith(text[:200])
                or text.startswith(stored_chunk.text[:200])
            ):
                return cid
        return None

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """Run full HydRAG pipeline and map results back to ScoredChunks."""
        retrieval_results = hydrag_search(
            adapter=self._store,
            query=query,
            n_results=n_results,
            config=self._config,
            llm=self._llm,
        )

        scored: list[ScoredChunk] = []
        for rr in retrieval_results:
            chunk_id = self._resolve_chunk_id(rr.text)
            if chunk_id is None:
                logger.debug("HeadHydrag: could not resolve chunk for text %.60s…", rr.text)
                continue
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            scored.append(ScoredChunk(
                chunk=chunk,
                score=rr.score,
                head_origin="head_hydrag",
            ))
        return scored

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "HeadHydrag":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
