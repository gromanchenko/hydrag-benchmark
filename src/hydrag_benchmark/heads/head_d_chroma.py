"""Head D (ChromaDB) — ChromaDB vector retrieval using Ollama embeddings.

Uses nomic-embed-text (or a configured model) via the Ollama /api/embed endpoint
to encode corpus chunks and query. All retrieval is by cosine similarity.

Requires: pip install chromadb>=0.4.0 (no chromadb embedding function plugin needed;
embeddings are computed via Ollama and passed as float vectors directly).

SurrealDB/SQLite heads use FTS (lexical); this head uses dense vector search.
The comparison across all three backends measures both lexical and semantic quality.
"""

from __future__ import annotations

import hashlib
import logging
import time

from .base import Chunk, ScoredChunk
from .head_ids import HEAD_ID_ALIASES

logger = logging.getLogger("hydrag_benchmark.heads.head_d_chroma")

# chromadb batch insert limit; stay well under the 41 666 hard cap.
_CHROMA_BATCH_SIZE = 2000

# Ollama embed batch size — keep small to avoid Ollama payload limits.
_EMBED_BATCH_SIZE = 8

# Max chars per text sent to Ollama — keeps tokens within model context window.
_MAX_EMBED_CHARS = 2000


class HeadDChroma:
    """ChromaDB vector retrieval head (BEIR-compatible).

    Index-time:
        All corpus chunks are embedded via Ollama and inserted into an
        in-memory ChromaDB collection (no disk persistence needed for BEIR).

    Query-time:
        The query text is embedded and matched by cosine similarity.
        Returns up to n_results chunks ranked by distance ascending.

    The Ollama host and embedding model are operator-configurable so the same
    head works both in CI (mocked / hash-embed fallback) and on the g6.2xlarge
    with real nomic-embed-text pulled into the AMI.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        ollama_timeout_s: float = 30.0,
        collection_name: str | None = None,
    ) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for HeadDChroma. "
                "Install with: pip install 'chromadb>=0.4.0,<0.5'"
            ) from exc

        self._ollama_host = ollama_host.rstrip("/")
        self._embedding_model = embedding_model
        self._ollama_timeout_s = ollama_timeout_s
        self._collection_name = collection_name or f"beir_{hashlib.sha1(embedding_model.encode()).hexdigest()[:8]}"
        self._chunks: dict[str, Chunk] = {}
        self._client: object | None = None
        self._collection: object | None = None

    @property
    def name(self) -> str:
        return HEAD_ID_ALIASES["chroma_vector"]

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed and return float vectors, one per text."""
        import json
        import urllib.error
        import urllib.request

        # Ollama returns HTTP 400 for empty strings; replace with whitespace.
        # Truncate long texts to stay within model context window.
        safe_texts = [
            (t[:_MAX_EMBED_CHARS] if len(t) > _MAX_EMBED_CHARS else t)
            if t.strip() else " "
            for t in texts
        ]
        payload = json.dumps(
            {"model": self._embedding_model, "input": safe_texts}
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self._ollama_host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._ollama_timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Ollama embed HTTP {exc.code} ({self._ollama_host}): {err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embed call failed ({self._ollama_host}): {exc}"
            ) from exc
        elapsed = time.monotonic() - t0

        if "embeddings" in body:
            vecs = body["embeddings"]
        elif "embedding" in body:
            # Legacy single-input path.
            vecs = [body["embedding"]]
        else:
            raise RuntimeError(f"Ollama response missing 'embeddings' key: {body!r}")

        logger.debug(
            "Embedded %d texts via %s (%.2fs)",
            len(texts),
            self._embedding_model,
            elapsed,
        )
        return [list(map(float, v)) for v in vecs]

    def _embed_single(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    # ── Index ─────────────────────────────────────────────────────────────────

    def build_index(self, chunks: list[Chunk]) -> None:
        """Encode all corpus chunks and insert into an in-memory ChromaDB collection."""
        import chromadb

        self._chunks = {c.chunk_id: c for c in chunks}

        # In-memory client: avoids disk I/O, safe for BEIR corpus sizes up to ~200 k chunks.
        self._client = chromadb.Client()
        try:
            self._client.delete_collection(self._collection_name)  # type: ignore[union-attr]
        except Exception:
            pass
        self._collection = self._client.create_collection(  # type: ignore[union-attr]
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        total = len(chunks)
        logger.info("HeadDChroma: embedding %d chunks via %s", total, self._embedding_model)

        ids: list[str] = []
        documents: list[str] = []
        embeddings_buf: list[list[float]] = []

        for i, chunk in enumerate(chunks):
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            if len(ids) == _CHROMA_BATCH_SIZE or i == total - 1:
                # Embed in sub-batches to stay within Ollama limits.
                all_vecs: list[list[float]] = []
                for eb_start in range(0, len(documents), _EMBED_BATCH_SIZE):
                    eb_slice = documents[eb_start : eb_start + _EMBED_BATCH_SIZE]
                    all_vecs.extend(self._embed_batch(eb_slice))
                embeddings_buf.extend(all_vecs)
                self._collection.add(  # type: ignore[union-attr]
                    ids=ids,
                    documents=documents,
                    embeddings=all_vecs,
                )
                logger.info(
                    "HeadDChroma: indexed batch %d–%d / %d",
                    i + 1 - len(ids) + 1,
                    i + 1,
                    total,
                )
                ids = []
                documents = []

        logger.info("HeadDChroma: index complete (%d chunks)", total)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """Cosine similarity search against the ChromaDB collection."""
        if self._collection is None:
            raise RuntimeError("HeadDChroma: build_index() must be called before retrieve()")

        query_vec = self._embed_single(query)
        raw = self._collection.query(  # type: ignore[union-attr]
            query_embeddings=[query_vec],
            n_results=min(n_results, len(self._chunks)),
            include=["distances"],
        )

        results: list[ScoredChunk] = []
        for chunk_id, distance in zip(
            raw["ids"][0],
            raw["distances"][0],
        ):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            # Cosine distance ∈ [0, 2]; convert to descending similarity score.
            similarity = 1.0 - (distance / 2.0)
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=similarity,
                    head_origin=self.name,
                )
            )
        # Already ordered ascending by distance → descending by similarity.
        return results

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            if self._client is not None and self._collection is not None:
                self._client.delete_collection(self._collection_name)  # type: ignore[union-attr]
        except Exception:
            pass
        self._client = None
        self._collection = None
