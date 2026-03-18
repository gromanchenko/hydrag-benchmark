"""Two-stage quality filter for Doc2Query generated questions.

Per RFC §4.2 steps 4 and 6:

Stage 1 — Lexical (no GPU needed):
  - Drop questions < 8 tokens (too vague)
  - Drop questions containing only chunk-internal variable names
  - Drop exact duplicates of chunk text substrings

Stage 2 — Semantic (after embedding):
  - Drop questions with cosine similarity < 0.3 to parent chunk embedding
"""

from __future__ import annotations

import math
import re


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


# ── Stage 1: Lexical filter ──────────────────────────────────────────────────


def lexical_filter(
    questions: list[str],
    chunk_text: str,
    min_tokens: int = 8,
) -> list[str]:
    """Stage 1 lexical filter. Returns surviving questions.

    Drops:
    - Questions with fewer than min_tokens tokens
    - Questions whose tokens are all chunk-internal identifiers
    - Questions that are exact substrings of the chunk text
    """
    chunk_lower = chunk_text.lower()
    chunk_identifiers = set(re.findall(r"[A-Za-z_]\w+", chunk_text.lower()))
    survivors: list[str] = []

    for q in questions:
        q_stripped = q.strip()
        if not q_stripped:
            continue

        tokens = _tokenize(q_stripped)

        # Drop < min_tokens
        if len(tokens) < min_tokens:
            continue

        # Drop if ALL tokens are chunk-internal identifiers
        non_ident_tokens = [t for t in tokens if t not in chunk_identifiers]
        if not non_ident_tokens:
            continue

        # Drop exact substring matches
        if q_stripped.lower() in chunk_lower:
            continue

        survivors.append(q_stripped)

    return survivors


# ── Stage 2: Semantic filter ─────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_filter(
    questions: list[str],
    question_embeddings: list[list[float]],
    chunk_embedding: list[float],
    min_similarity: float = 0.3,
) -> list[str]:
    """Stage 2 semantic filter. Returns questions with cosine sim >= threshold.

    Requires pre-computed embeddings for both questions and parent chunk.
    """
    survivors: list[str] = []
    for q, q_emb in zip(questions, question_embeddings):
        sim = cosine_similarity(q_emb, chunk_embedding)
        if sim >= min_similarity:
            survivors.append(q)
    return survivors
