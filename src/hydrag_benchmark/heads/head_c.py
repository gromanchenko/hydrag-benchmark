"""Head C — Hybrid retrieval: Head A candidates reranked by Head B embeddings.

Per RFC §5:
1. Run Head A to produce top-N candidates by structural score
2. Look up each candidate's pre-computed dense embedding from Head B
3. Compute cosine similarity between query embedding and candidate's primary vector
4. Rerank by α × structural_score + (1 - α) × cosine_similarity
"""

from __future__ import annotations

from .base import Chunk, ScoredChunk
from .head_a import HeadA
from .head_b import HeadB
from ..quality_filter import cosine_similarity


class HeadC:
    """Hybrid retrieval: graph candidates + cosine rerank.

    No LLM at query time. One embedding call for the query only.
    Reranking is a dot product over the candidate set.
    """

    def __init__(
        self,
        head_a: HeadA,
        head_b: HeadB,
        alpha: float = 0.5,
        candidate_pool: int = 50,
    ) -> None:
        self._head_a = head_a
        self._head_b = head_b
        self._alpha = alpha
        self._candidate_pool = candidate_pool

    @property
    def name(self) -> str:
        return "head_c"

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """Hybrid retrieval: Head A candidates reranked by Head B cosine sim."""
        # Step 1: Get structural candidates from Head A
        candidates = self._head_a.retrieve(query, n_results=self._candidate_pool)
        if not candidates:
            return []

        # Step 2: Embed the query
        query_emb = self._head_b._embedder.embed([query])[0]

        # Step 3: Rerank candidates
        reranked: list[tuple[ScoredChunk, float]] = []
        max_structural = max(c.score for c in candidates) if candidates else 1.0
        if max_structural == 0.0:
            max_structural = 1.0

        for candidate in candidates:
            # Normalize structural score to [0, 1]
            structural_norm = candidate.score / max_structural

            # Look up pre-computed embedding from Head B
            chunk_emb = self._head_b.get_chunk_embedding(candidate.chunk.chunk_id)
            if chunk_emb is None:
                # No embedding available — use normalized structural score as fallback
                combined = structural_norm
            else:
                cos_sim = cosine_similarity(query_emb, chunk_emb)
                # Clamp to [0, 1] for combination
                cos_sim = max(0.0, min(1.0, cos_sim))
                combined = self._alpha * structural_norm + (1.0 - self._alpha) * cos_sim

            reranked.append((candidate, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for scored, combined_score in reranked[:n_results]:
            results.append(ScoredChunk(
                chunk=scored.chunk,
                score=combined_score,
                head_origin="head_c",
            ))
        return results
