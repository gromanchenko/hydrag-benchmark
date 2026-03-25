"""Tests for Head B pipeline, augmentation cache, embedding, and Doc2Query."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydrag_benchmark.augmentation_cache import AugmentationCache, CacheEntry
from hydrag_benchmark.doc2query import Doc2QueryGenerator
from hydrag_benchmark.embedding import HashEmbedder
from hydrag_benchmark.heads.base import Chunk
from hydrag_benchmark.heads.head_b import HeadB, HeadBIndex, VectorEntry


# ── HashEmbedder ─────────────────────────────────────────────────────────────


class TestHashEmbedder:
    def test_dimension(self) -> None:
        e = HashEmbedder(dimension=128)
        assert e.dimension == 128

    def test_output_shape(self) -> None:
        e = HashEmbedder(dimension=64)
        result = e.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 64
        assert len(result[1]) == 64

    def test_deterministic(self) -> None:
        e = HashEmbedder(dimension=64)
        r1 = e.embed(["test"])
        r2 = e.embed(["test"])
        assert r1 == r2

    def test_different_text_different_vectors(self) -> None:
        e = HashEmbedder(dimension=64)
        r = e.embed(["alpha", "beta"])
        assert r[0] != r[1]

    def test_unit_norm(self) -> None:
        import math
        e = HashEmbedder(dimension=128)
        v = e.embed(["normalize me"])[0]
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, abs=1e-6)


# ── AugmentationCache ────────────────────────────────────────────────────────


class TestAugmentationCache:
    def test_empty_cache(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json")
        assert cache.should_process("abc123") is True
        assert cache.get("abc123") is None

    def test_mark_success_skips(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json")
        cache.mark_success("abc123", ["Q1", "Q2"])
        assert cache.should_process("abc123") is False
        entry = cache.get("abc123")
        assert entry is not None
        assert entry.status == "success"
        assert entry.questions == ["Q1", "Q2"]

    def test_mark_failed_retries(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json", max_retries=3)
        cache.mark_failed("abc123")
        assert cache.should_process("abc123") is True  # attempt 1/3
        cache.mark_failed("abc123")
        assert cache.should_process("abc123") is True  # attempt 2/3
        cache.mark_failed("abc123")
        assert cache.should_process("abc123") is False  # attempt 3/3, exhausted

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        c1 = AugmentationCache(path)
        c1.mark_success("hash1", ["Q1"])
        c1.save()

        c2 = AugmentationCache(path)
        assert c2.should_process("hash1") is False
        entry = c2.get("hash1")
        assert entry is not None
        assert entry.questions == ["Q1"]

    def test_stats(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json")
        cache.mark_success("a", ["Q"])
        cache.mark_failed("b")
        stats = cache.stats
        assert stats["success"] == 1
        assert stats["failed"] == 1
        assert stats["total"] == 2


# ── Doc2QueryGenerator (parse only, no LLM call) ────────────────────────────


class TestDoc2QueryParsing:
    def test_parse_numbered_questions(self) -> None:
        text = "1. How does fibonacci work?\n2. Why does the function fail when n is negative?\n3. What is the time complexity?"
        questions = Doc2QueryGenerator._parse_questions(text)
        assert len(questions) == 3
        assert "How does fibonacci work?" in questions

    def test_parse_bullet_questions(self) -> None:
        text = "- How does fibonacci work?\n- Why does the function return None?"
        questions = Doc2QueryGenerator._parse_questions(text)
        assert len(questions) == 2

    def test_parse_plain_questions(self) -> None:
        text = "How does fibonacci work?\nWhy does the function fail when n is negative?"
        questions = Doc2QueryGenerator._parse_questions(text)
        assert len(questions) == 2

    def test_filters_short_lines(self) -> None:
        text = "OK\nHow does the fibonacci function calculate the nth number?"
        questions = Doc2QueryGenerator._parse_questions(text)
        assert len(questions) == 1


# ── HeadB integration ────────────────────────────────────────────────────────


class _FakeDoc2Query:
    """Fake Doc2Query that returns canned questions."""

    def generate(self, chunk_text: str) -> list[str]:
        return [
            f"How does this chunk handle the processing of data in the system?",
            f"Why does this implementation fail when the input is empty or null?",
            f"What is the algorithmic complexity of the main operation described here?",
            f"short",  # Should be filtered by lexical filter
            chunk_text[:50],  # Exact substring — should be filtered
        ]


@pytest.fixture()
def head_b_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="ch-b1", source="algo.py",
            text="def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
        ),
        Chunk(
            chunk_id="ch-b2", source="io.py",
            text="def read_csv(path):\n    import csv\n    with open(path) as f:\n        reader = csv.reader(f)\n        return list(reader)",
        ),
    ]


class TestHeadB:
    def test_build_index(self, head_b_chunks: list[Chunk]) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
        idx = head.build_index(head_b_chunks)
        assert len(idx.chunks) == 2
        assert len(idx.entries) > 0  # At least primary vectors

    def test_retrieve_returns_results(self, head_b_chunks: list[Chunk]) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
        head.build_index(head_b_chunks)
        results = head.retrieve("quicksort algorithm", n_results=5)
        assert len(results) > 0

    def test_retrieve_ordered_by_score(self, head_b_chunks: list[Chunk]) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
        head.build_index(head_b_chunks)
        results = head.retrieve("sorting algorithm", n_results=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_head_name(self, head_b_chunks: list[Chunk]) -> None:
        head = HeadB(embedder=HashEmbedder(dimension=64))
        assert head.name == "head_b"

    def test_filter_stats(self, head_b_chunks: list[Chunk]) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
        head.build_index(head_b_chunks)
        stats = head.index.filter_stats
        assert stats["total_generated"] == 10  # 5 questions × 2 chunks
        assert stats["lexical_rejected"] > 0  # "short" gets filtered

    def test_filter_rejection_rate(self, head_b_chunks: list[Chunk]) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
        head.build_index(head_b_chunks)
        rate = head.filter_rejection_rate
        assert rate > 0.0  # Some questions should be rejected

    def test_sidecar_json(self, head_b_chunks: list[Chunk], tmp_path: Path) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
        head.build_index(head_b_chunks)
        sidecar = tmp_path / "sidecar.json"
        head.save_sidecar(sidecar)
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert isinstance(data, dict)
        assert "ch-b1" in data

    def test_get_chunk_embedding(self, head_b_chunks: list[Chunk]) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
        head.build_index(head_b_chunks)
        emb = head.get_chunk_embedding("ch-b1")
        assert emb is not None
        assert len(emb) == 64

    def test_no_index_raises(self) -> None:
        head = HeadB(embedder=HashEmbedder(dimension=64))
        with pytest.raises(RuntimeError, match="index not built"):
            head.retrieve("test")

    def test_set_index(self, head_b_chunks: list[Chunk]) -> None:
        embedder = HashEmbedder(dimension=64)
        head = HeadB(embedder=embedder)
        idx = HeadBIndex()
        idx.chunks = {c.chunk_id: c for c in head_b_chunks}
        embs = embedder.embed([c.text for c in head_b_chunks])
        for chunk, emb in zip(head_b_chunks, embs):
            idx.entries.append(VectorEntry(
                vector=emb, chunk_id=chunk.chunk_id,
                is_primary=True, text=chunk.text,
            ))
        head.set_index(idx)
        results = head.retrieve("quicksort", n_results=5)
        assert len(results) > 0

    def test_cache_integration(self, head_b_chunks: list[Chunk], tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        embedder = HashEmbedder(dimension=64)
        head = HeadB(
            embedder=embedder,
            doc2query=_FakeDoc2Query(),
            cache_path=cache_path,
        )
        head.build_index(head_b_chunks)
        assert cache_path.exists()
