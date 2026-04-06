"""Canonical head IDs and compatibility aliases for benchmark heads."""

from __future__ import annotations

CANONICAL_HEAD_IDS: tuple[str, ...] = (
    "symbol_graph",
    "dense_doc2query",
    "hybrid_graph_dense",
    "fts5_baseline",
    "fts5_enriched",
    "hydrag_full",
    # T-964: multi-backend comparison heads
    "surreal_fts",    # SurrealDB disjunctive FTS (HeadDSurreal)
    "chroma_vector",  # ChromaDB cosine-similarity vector search (HeadDChroma)
)

HEAD_ID_ALIASES: dict[str, str] = {
    # Canonical IDs
    "symbol_graph": "symbol_graph",
    "dense_doc2query": "dense_doc2query",
    "hybrid_graph_dense": "hybrid_graph_dense",
    "fts5_baseline": "fts5_baseline",
    "fts5_enriched": "fts5_enriched",
    "hydrag_full": "hydrag_full",
    "surreal_fts": "surreal_fts",
    "chroma_vector": "chroma_vector",
    # Legacy IDs (kept for backward compatibility)
    "head_a": "symbol_graph",
    "head_b": "dense_doc2query",
    "head_c": "hybrid_graph_dense",
    "head_d": "fts5_baseline",
    "head_e": "fts5_enriched",
    "head_hydrag": "hydrag_full",
}


def normalize_head_id(head_id: str) -> str | None:
    """Return canonical head ID for canonical or legacy input; None if unknown."""
    return HEAD_ID_ALIASES.get(head_id.strip().lower())


def canonicalize_head_ids(head_ids: list[str]) -> list[str]:
    """Normalize a head-id list to canonical IDs while preserving order and uniqueness."""
    canonical: list[str] = []
    seen: set[str] = set()
    for head_id in head_ids:
        normalized = normalize_head_id(head_id)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        canonical.append(normalized)
    return canonical
