"""Multi-head retrieval benchmark — Heads A–E + HydRAG."""

from .base import Chunk, Embedder, RetrievalHead, ScoredChunk
from .head_a import HeadA
from .head_b import HeadB
from .head_c import HeadC
from .head_d import HeadD
from .head_e import HeadE
from .head_hydrag import HeadHydrag
from .head_ids import CANONICAL_HEAD_IDS, HEAD_ID_ALIASES, canonicalize_head_ids, normalize_head_id

__all__ = [
    "Chunk",
    "CANONICAL_HEAD_IDS",
    "HEAD_ID_ALIASES",
    "Embedder",
    "HeadA",
    "HeadB",
    "HeadC",
    "HeadD",
    "HeadE",
    "HeadHydrag",
    "RetrievalHead",
    "ScoredChunk",
    "canonicalize_head_ids",
    "normalize_head_id",
]
