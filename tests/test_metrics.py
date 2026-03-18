"""Tests for hydrag_benchmark.metrics — frozen 0.1 metric set."""

from hydrag_benchmark.metrics import (
    chunk_overlap,
    latency_stats,
    mrr,
    recall_at_1,
    recall_at_k,
)


class TestRecallAt1:
    def test_hit(self) -> None:
        assert recall_at_1(["contains fibonacci and more"], ["fibonacci"]) == 1.0

    def test_miss(self) -> None:
        assert recall_at_1(["unrelated content"], ["fibonacci"]) == 0.0

    def test_empty_retrieved(self) -> None:
        assert recall_at_1([], ["fibonacci"]) == 0.0

    def test_empty_phrases(self) -> None:
        assert recall_at_1(["anything"], []) == 0.0

    def test_case_insensitive(self) -> None:
        assert recall_at_1(["FIBONACCI numbers"], ["fibonacci"]) == 1.0


class TestRecallAtK:
    def test_all_found(self) -> None:
        retrieved = ["chunk with fibonacci", "chunk with numbers"]
        assert recall_at_k(retrieved, ["fibonacci", "numbers"]) == 1.0

    def test_partial(self) -> None:
        retrieved = ["chunk with fibonacci", "unrelated"]
        assert recall_at_k(retrieved, ["fibonacci", "numbers"]) == 0.5

    def test_none_found(self) -> None:
        assert recall_at_k(["unrelated"], ["fibonacci", "numbers"]) == 0.0

    def test_empty_phrases(self) -> None:
        assert recall_at_k(["anything"], []) == 1.0


class TestMRR:
    def test_first_rank(self) -> None:
        assert mrr(["fibonacci impl", "other"], ["fibonacci"]) == 1.0

    def test_second_rank(self) -> None:
        assert mrr(["other", "fibonacci impl"], ["fibonacci"]) == 0.5

    def test_not_found(self) -> None:
        assert mrr(["other", "another"], ["fibonacci"]) == 0.0


class TestChunkOverlap:
    def test_full_overlap(self) -> None:
        retrieved = ["the quick brown fox"]
        result = chunk_overlap(retrieved, ["quick brown"])
        assert result == 1.0

    def test_partial_overlap(self) -> None:
        retrieved = ["the quick red fox"]
        result = chunk_overlap(retrieved, ["quick brown"])
        assert result == 0.5

    def test_no_overlap(self) -> None:
        assert chunk_overlap(["unrelated text"], ["fibonacci numbers"]) == 0.0

    def test_empty(self) -> None:
        assert chunk_overlap([], ["fibonacci"]) == 0.0


class TestLatencyStats:
    def test_basic(self) -> None:
        stats = latency_stats([10.0, 20.0, 30.0, 40.0, 50.0])
        assert stats["avg"] == 30.0
        assert stats["p50"] == 30.0
        assert stats["p95"] > 40.0
        assert stats["p99"] > 45.0

    def test_single(self) -> None:
        stats = latency_stats([42.0])
        assert stats["avg"] == 42.0
        assert stats["p50"] == 42.0
        assert stats["p95"] == 42.0

    def test_empty(self) -> None:
        stats = latency_stats([])
        assert stats == {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
