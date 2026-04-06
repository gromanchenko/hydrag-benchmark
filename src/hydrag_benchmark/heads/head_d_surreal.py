"""Head D (SurrealDB) — SurrealDB disjunctive FTS retrieval. Zero GPU, zero dense model.

Wraps hydrag-core's SurrealDBAdapter for BEIR benchmarking using the disjunctive
FTS query path (@@-based, term OR) that produces non-zero retrieval quality.
Requires a running SurrealDB instance (default: ws://localhost:8000/rpc).
Requires: pip install hydrag-core[surrealdb]
"""

from __future__ import annotations

import logging
import uuid

from hydrag import IndexedChunk
from hydrag.surreal_adapter import SurrealDBAdapter

from .base import Chunk, ScoredChunk
from .head_ids import HEAD_ID_ALIASES

logger = logging.getLogger("hydrag_benchmark.heads.head_d_surreal")

# Dummy 1-dim embed function: FTS path in SurrealDBAdapter does not use embeddings.
# embedding_dim must be a positive integer per SurrealDBAdapter contract.
_DUMMY_EMBED: list[float] = [0.0]


def _dummy_embed_fn(text: str) -> list[float]:  # noqa: ARG001
    return _DUMMY_EMBED


class HeadDSurreal:
    """SurrealDB FTS head — lexical retrieval via disjunctive @@ operator.

    Equivalent access pattern to HeadD (SQLite FTS5) but backed by SurrealDB.
    Uses keyword_search() which internally calls _build_disjunctive_fts_query().

    The database name is auto-generated per instance to isolate concurrent runs.
    Call close() when done to release the connection thread.
    """

    def __init__(
        self,
        surrealdb_url: str = "ws://localhost:8000/rpc",
        namespace: str = "hydrag_beir",
        database: str | None = None,
        timeout: int = 300,
        username: str | None = None,
        password: str | None = None,
        batch_size: int = 2000,
    ) -> None:
        run_token = uuid.uuid4().hex[:8]
        db_name = database or f"beir_{run_token}"
        self._adapter = SurrealDBAdapter(
            url=surrealdb_url,
            embedding_dim=1,
            embed_fn=_dummy_embed_fn,
            namespace=namespace,
            database=db_name,
            timeout=timeout,
            auto_schema=True,
            username=username,
            password=password,
            allow_insecure_auth=True,  # localhost-only per S-003
            batch_size=batch_size,
            assume_fresh=True,
            deferred_index=True,
            fts_fields=["raw_content"],
        )
        self._adapter._connect()
        logger.info(
            "HeadDSurreal connected to %s ns=%s db=%s (batch=%d, deferred_index)",
            surrealdb_url,
            namespace,
            db_name,
            batch_size,
        )

    @property
    def name(self) -> str:
        return HEAD_ID_ALIASES["surreal_fts"]

    def build_index(self, chunks: list[Chunk]) -> None:
        """Index chunks into SurrealDB via batch insert."""
        indexed_chunks: list[IndexedChunk] = [
            IndexedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title="",
                raw_content=chunk.text,
                content_hash="",  # auto-filled by adapter
            )
            for chunk in chunks
        ]
        created = self._adapter.index_documents(indexed_chunks, embeddings=None)
        logger.info("HeadDSurreal indexed %d chunks (SurrealDB FTS)", created)

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """Disjunctive FTS retrieval via SurrealDB @@ operator."""
        raw_rows = self._adapter._keyword_search_with_ids(
            query, n_results=n_results,
        )
        results: list[ScoredChunk] = []
        for rank, row in enumerate(raw_rows):
            results.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=row["chunk_id"],
                        source=row["source"],
                        text=row["raw_content"],
                    ),
                    score=1.0 / (rank + 1),
                    head_origin=self.name,
                )
            )
        return results

    def close(self) -> None:
        try:
            self._adapter.close()
        except Exception:
            pass
