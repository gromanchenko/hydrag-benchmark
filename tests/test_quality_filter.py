"""Tests for quality filter — lexical and semantic stages."""

from __future__ import annotations

import pytest

from hydrag_benchmark.quality_filter import (
    cosine_similarity,
    lexical_filter,
    semantic_filter,
)


# ── Cosine similarity ────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert cosine_similarity(a, b) == 0.0


# ── Lexical filter ───────────────────────────────────────────────────────────


class TestLexicalFilter:
    def test_drops_short_questions(self) -> None:
        questions = [
            "What is it?",  # 3 tokens
            "How does the fibonacci function calculate the nth number in the sequence?",  # 13 tokens
        ]
        result = lexical_filter(questions, "fibonacci function code")
        assert len(result) == 1
        assert "fibonacci" in result[0]

    def test_drops_chunk_internal_only(self) -> None:
        chunk = "def fibonacci(n):\n    return fibonacci(n-1) + fibonacci(n-2)"
        questions = [
            "fibonacci fibonacci fibonacci fibonacci fibonacci fibonacci fibonacci fibonacci",  # all chunk-internal
        ]
        result = lexical_filter(questions, chunk)
        assert len(result) == 0

    def test_drops_exact_substring(self) -> None:
        chunk = "The fibonacci function calculates numbers in the sequence recursively"
        questions = [
            "The fibonacci function calculates numbers in the sequence recursively",
        ]
        result = lexical_filter(questions, chunk)
        assert len(result) == 0

    def test_keeps_valid_questions(self) -> None:
        chunk = "def fibonacci(n):\n    if n <= 1:\n        return n"
        questions = [
            "How does the fibonacci function handle the base case when n is less than or equal to one?",
        ]
        result = lexical_filter(questions, chunk)
        assert len(result) == 1

    def test_empty_input(self) -> None:
        assert lexical_filter([], "some chunk") == []
        assert lexical_filter([""], "some chunk") == []

    def test_custom_min_tokens(self) -> None:
        questions = ["What does this function do when called with four arguments?"]
        assert len(lexical_filter(questions, "code", min_tokens=4)) == 1
        assert len(lexical_filter(questions, "code", min_tokens=20)) == 0


# ── Semantic filter ──────────────────────────────────────────────────────────


class TestSemanticFilter:
    def test_keeps_similar(self) -> None:
        questions = ["How does fibonacci work?"]
        q_embs = [[0.9, 0.1, 0.0]]
        chunk_emb = [1.0, 0.0, 0.0]
        result = semantic_filter(questions, q_embs, chunk_emb, min_similarity=0.3)
        assert len(result) == 1

    def test_drops_dissimilar(self) -> None:
        questions = ["How does fibonacci work?"]
        q_embs = [[0.0, 1.0, 0.0]]
        chunk_emb = [1.0, 0.0, 0.0]
        result = semantic_filter(questions, q_embs, chunk_emb, min_similarity=0.3)
        assert len(result) == 0

    def test_mixed_filter(self) -> None:
        questions = ["Good question", "Bad question"]
        q_embs = [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
        chunk_emb = [1.0, 0.0, 0.0]
        result = semantic_filter(questions, q_embs, chunk_emb, min_similarity=0.3)
        assert result == ["Good question"]
