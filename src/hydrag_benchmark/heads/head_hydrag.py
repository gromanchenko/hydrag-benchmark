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
import json
import logging
from pathlib import Path
from typing import Callable, Optional, Protocol
from urllib import request

from hydrag.config import HydRAGConfig
from hydrag.core import hydrag_search
from hydrag.protocols import LLMProvider, VectorStoreAdapter
from hydrag.sqlite_store import IndexedChunk, SQLiteFTSStore

from .base import Chunk, ScoredChunk
from .head_ids import HEAD_ID_ALIASES

logger = logging.getLogger("hydrag_benchmark.heads.head_hydrag")


def _text_hash(text: str) -> str:
    """Stable hash for reverse-index key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ollama_embed_fn(host: str, model: str) -> Callable[[str], list[float]]:
    """Build a small sync embedding function against the Ollama HTTP API."""

    endpoint = f"{host.rstrip('/')}/api/embeddings"

    def _embed(text: str) -> list[float]:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))

        embedding = data.get("embedding")
        if isinstance(embedding, list) and embedding:
            return [float(x) for x in embedding]

        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return [float(x) for x in embeddings[0]]

        raise RuntimeError("Ollama embedding response missing vector")

    return _embed


class _IndexedVectorStoreAdapter(VectorStoreAdapter, Protocol):
    """Local extension of hydrag's VectorStoreAdapter (T-5061): adds the
    index_documents/close lifecycle methods both SQLiteFTSStore and
    SurrealDBAdapter implement but that hydrag-core's own protocol
    deliberately omits (it only covers query-time search). Defined here
    rather than in hydrag-core, which is out of scope for this ticket."""

    def index_documents(self, chunks: list[IndexedChunk]) -> int: ...

    def close(self) -> None: ...


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
        db_backend: str = "sqlite",
        enable_crag: bool = True,
        profile: str = "prose",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "qwen3:4b",
        surrealdb_url: str = "ws://localhost:8000/rpc",
        surrealdb_namespace: str = "hydrag_benchmark",
        surrealdb_database: str = "default",
        surrealdb_username: str | None = None,
        surrealdb_password: str | None = None,
        surrealdb_token: str | None = None,
        surrealdb_embedding_model: str = "nomic-embed-text",
        surrealdb_embedding_dim: int = 768,
        surrealdb_allow_insecure_auth: bool = False,
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self._backend = db_backend.strip().lower()
        # Explicit protocol annotation: both branches below satisfy
        # VectorStoreAdapter structurally, but are different concrete
        # classes (T-5061, same pattern as beir_runner.py's head dispatch).
        self._store: _IndexedVectorStoreAdapter
        if self._backend == "sqlite":
            self._store = SQLiteFTSStore(db_path)
        elif self._backend == "surrealdb":
            from hydrag.surreal_adapter import SurrealDBAdapter

            # Surreal adapter requires an explicit connect lifecycle.
            self._store = SurrealDBAdapter(
                url=surrealdb_url,
                embedding_dim=surrealdb_embedding_dim,
                embed_fn=_ollama_embed_fn(ollama_host, surrealdb_embedding_model),
                namespace=surrealdb_namespace,
                database=surrealdb_database,
                username=surrealdb_username,
                password=surrealdb_password,
                token=surrealdb_token,
                allow_insecure_auth=surrealdb_allow_insecure_auth,
            ).__enter__()
        else:
            raise ValueError("db_backend must be one of: sqlite, surrealdb")

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
        return HEAD_ID_ALIASES["head_hydrag"]

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
        logger.info("HeadHydrag indexed %d chunks (backend=%s)", count, self._backend)

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
                head_origin=self.name,
                # B-05: preserve which internal hydrag-core head actually
                # produced this result, and its fast-path/crag-skip flags
                # -- otherwise every result is indistinguishably labeled
                # with this benchmark head's own static name, and a
                # BM25-only run looks identical to a full-pipeline run.
                metadata={
                    "hydrag_head_origin": rr.head_origin,
                    "fast_path_triggered": bool(rr.metadata.get("fast_path_triggered", False)),
                    "crag_skipped": bool(rr.metadata.get("crag_skipped", False)),
                },
            ))
        return scored

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "HeadHydrag":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
