"""Head B — Doc2Query + Dense Embedding retrieval.

Index-time: For each chunk, generate synthetic developer questions via an
instruct SLM, filter with two-stage quality gate, embed chunk + surviving
questions into dual-vector storage (primary + secondary vectors).

Query-time: Embed the query, compute cosine similarity against all primary
and secondary vectors, return parent chunks ranked by max similarity.

Per RFC §4: storage is dual-vector (not concatenation) to avoid false-positive
substring matches in Studio's runtime Head 1a ChromaDB $contains filters.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from .base import Chunk, Embedder, ScoredChunk
from ..augmentation_cache import AugmentationCache
from ..doc2query import Doc2QueryConfig, Doc2QueryGenerator
from ..quality_filter import cosine_similarity, lexical_filter, semantic_filter

logger = logging.getLogger("hydrag_benchmark.heads.head_b")


@dataclass
class VectorEntry:
    """A single vector in the dual-vector store."""

    vector: list[float]
    chunk_id: str
    is_primary: bool
    text: str  # chunk text for primary, question text for secondary


@dataclass
class HeadBIndex:
    """Pre-built index for Head B retrieval."""

    entries: list[VectorEntry] = field(default_factory=list)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    sidecar: dict[str, list[str]] = field(default_factory=dict)  # chunk_id → questions
    filter_stats: dict[str, int] = field(default_factory=lambda: {
        "total_generated": 0,
        "lexical_rejected": 0,
        "semantic_rejected": 0,
        "survived": 0,
    })


class HeadB:
    """Doc2Query + Dense Embedding retrieval head.

    Two-phase lifecycle:
    1. build_index() — index-time: generates questions, filters, embeds
    2. retrieve() — query-time: embeds query, cosine search against index
    """

    def __init__(
        self,
        embedder: Embedder,
        doc2query: Doc2QueryGenerator | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._embedder = embedder
        self._doc2query = doc2query or Doc2QueryGenerator()
        self._cache_path = cache_path
        self._index: HeadBIndex | None = None

    @property
    def name(self) -> str:
        return "head_b"

    @property
    def index(self) -> HeadBIndex | None:
        return self._index

    def build_index(self, chunks: list[Chunk]) -> HeadBIndex:
        """Index-time pipeline: Doc2Query → filter → embed → store.

        Per RFC §4.2 pipeline steps 1–8.
        """
        idx = HeadBIndex()
        idx.chunks = {c.chunk_id: c for c in chunks}

        cache = AugmentationCache(self._cache_path) if self._cache_path else None
        config_fp = getattr(self._doc2query, 'config_fingerprint', '')

        # Step 1: Generate synthetic questions per chunk
        all_questions: dict[str, list[str]] = {}
        for chunk in chunks:
            chunk_hash = Chunk.content_address(chunk.text)
            cache_key = f"{chunk_hash}:{config_fp}" if config_fp else chunk_hash

            if cache and not cache.should_process(cache_key):
                entry = cache.get(cache_key)
                if entry and entry.status == "success":
                    all_questions[chunk.chunk_id] = entry.questions
                continue

            try:
                questions = self._doc2query.generate(chunk.text)
                if cache:
                    cache.mark_success(cache_key, questions)
                all_questions[chunk.chunk_id] = questions
            except Exception:
                logger.warning("Doc2Query failed for chunk %s", chunk.chunk_id)
                if cache:
                    cache.mark_failed(cache_key)
                all_questions[chunk.chunk_id] = []

        if cache:
            cache.save()

        # Step 2: Lexical filter (Stage 1)
        filtered_questions: dict[str, list[str]] = {}
        for chunk_id, questions in all_questions.items():
            idx.filter_stats["total_generated"] += len(questions)
            chunk = idx.chunks[chunk_id]
            surviving = lexical_filter(questions, chunk.text)
            idx.filter_stats["lexical_rejected"] += len(questions) - len(surviving)
            filtered_questions[chunk_id] = surviving

        # Step 3: Embed chunks (primary vectors)
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self._embedder.embed(chunk_texts)

        chunk_emb_map: dict[str, list[float]] = {}
        for chunk, emb in zip(chunks, chunk_embeddings):
            idx.entries.append(VectorEntry(
                vector=emb, chunk_id=chunk.chunk_id,
                is_primary=True, text=chunk.text,
            ))
            chunk_emb_map[chunk.chunk_id] = emb

        # Step 4: Embed surviving questions + semantic filter (Stage 2)
        for chunk_id, questions in filtered_questions.items():
            if not questions:
                idx.sidecar[chunk_id] = []
                continue

            q_embeddings = self._embedder.embed(questions)
            chunk_emb = chunk_emb_map[chunk_id]

            # Semantic filter: keep only questions with cosine >= 0.3
            final_questions = semantic_filter(
                questions, q_embeddings, chunk_emb, min_similarity=0.3,
            )
            idx.filter_stats["semantic_rejected"] += len(questions) - len(final_questions)
            idx.filter_stats["survived"] += len(final_questions)

            idx.sidecar[chunk_id] = final_questions

            # Add secondary vectors for surviving questions
            for q, q_emb in zip(questions, q_embeddings):
                if q in final_questions:
                    idx.entries.append(VectorEntry(
                        vector=q_emb, chunk_id=chunk_id,
                        is_primary=False, text=q,
                    ))

        self._index = idx
        return idx

    def set_index(self, index: HeadBIndex) -> None:
        """Inject a pre-built index (for testing or deserialization)."""
        self._index = index

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """Query-time: embed query, cosine search, return ranked parent chunks."""
        if self._index is None:
            raise RuntimeError("HeadB index not built. Call build_index() first.")

        query_emb = self._embedder.embed([query])[0]

        # Compute cosine similarity against all vectors
        chunk_max_sim: dict[str, float] = {}
        for entry in self._index.entries:
            sim = cosine_similarity(query_emb, entry.vector)
            if entry.chunk_id not in chunk_max_sim or sim > chunk_max_sim[entry.chunk_id]:
                chunk_max_sim[entry.chunk_id] = sim

        # Sort by max similarity
        ranked = sorted(chunk_max_sim.items(), key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for chunk_id, score in ranked[:n_results]:
            chunk = self._index.chunks.get(chunk_id)
            if chunk:
                results.append(ScoredChunk(
                    chunk=chunk, score=score, head_origin="head_b",
                ))
        return results

    def get_chunk_embedding(self, chunk_id: str) -> list[float] | None:
        """Look up the primary embedding for a chunk (used by Head C)."""
        if self._index is None:
            return None
        for entry in self._index.entries:
            if entry.chunk_id == chunk_id and entry.is_primary:
                return entry.vector
        return None

    def save_sidecar(self, output_path: Path) -> None:
        """Persist generated questions as JSON sidecar per DoD item 5."""
        if self._index is None:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self._index.sidecar, indent=2), encoding="utf-8",
        )
        logger.info("Saved question sidecar to %s", output_path)

    @property
    def filter_rejection_rate(self) -> float:
        """Fraction of generated questions rejected by both filter stages."""
        if self._index is None:
            return 0.0
        total = self._index.filter_stats["total_generated"]
        if total == 0:
            return 0.0
        rejected = (
            self._index.filter_stats["lexical_rejected"]
            + self._index.filter_stats["semantic_rejected"]
        )
        return rejected / total
