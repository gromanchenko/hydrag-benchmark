"""Tests for HeadHydrag — full HydRAG pipeline as BEIR benchmark head."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hydrag_benchmark.heads.base import Chunk, ScoredChunk
from hydrag_benchmark.heads.head_hydrag import HeadHydrag, _text_hash


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="doc-1", text="Machine learning algorithms transform data into predictions.", source="beir:1"),
        Chunk(chunk_id="doc-2", text="Deep neural networks have revolutionized computer vision tasks.", source="beir:2"),
        Chunk(chunk_id="doc-3", text="Natural language processing enables text understanding.", source="beir:3"),
        Chunk(chunk_id="doc-4", text="Reinforcement learning agents learn through trial and error.", source="beir:4"),
        Chunk(chunk_id="doc-5", text="Support vector machines are effective for classification.", source="beir:5"),
    ]


# ── Basic Tests ──────────────────────────────────────────────────────────────


class TestHeadHydragBasic:
    def test_name(self) -> None:
        head = HeadHydrag()
        assert head.name == "head_hydrag"
        head.close()

    def test_context_manager(self) -> None:
        with HeadHydrag() as head:
            assert head.name == "head_hydrag"

    def test_build_index(self, sample_chunks: list[Chunk]) -> None:
        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            assert len(head._chunks) == 5
            assert len(head._text_to_chunk_id) == 5

    def test_text_hash_deterministic(self) -> None:
        h1 = _text_hash("hello world")
        h2 = _text_hash("hello world")
        h3 = _text_hash("different text")
        assert h1 == h2
        assert h1 != h3

    def test_reverse_index_maps_correctly(self, sample_chunks: list[Chunk]) -> None:
        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            for chunk in sample_chunks:
                resolved = head._resolve_chunk_id(chunk.text)
                assert resolved == chunk.chunk_id


# ── Retrieval Tests (mocked hydrag_search) ───────────────────────────────────


class TestHeadHydragRetrieval:
    def test_retrieve_maps_results_to_scored_chunks(self, sample_chunks: list[Chunk]) -> None:
        """Verify that hydrag_search results get mapped back to ScoredChunks with correct chunk_ids."""
        from hydrag.fusion import RetrievalResult

        mock_results = [
            RetrievalResult(
                text=sample_chunks[0].text,
                source="",
                score=0.9,
                head_origin="hydrag",
                trust_level="local",
            ),
            RetrievalResult(
                text=sample_chunks[2].text,
                source="",
                score=0.7,
                head_origin="hydrag",
                trust_level="local",
            ),
        ]

        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            with patch("hydrag_benchmark.heads.head_hydrag.hydrag_search", return_value=mock_results):
                results = head.retrieve("machine learning", n_results=5)

        assert len(results) == 2
        assert results[0].chunk.chunk_id == "doc-1"
        assert results[0].score == 0.9
        assert results[0].head_origin == "head_hydrag"
        assert results[1].chunk.chunk_id == "doc-3"
        assert results[1].score == 0.7

    def test_retrieve_skips_unresolvable_text(self, sample_chunks: list[Chunk]) -> None:
        """Results with text not in the index are silently dropped."""
        from hydrag.fusion import RetrievalResult

        mock_results = [
            RetrievalResult(
                text="this text was never indexed",
                source="",
                score=0.5,
                head_origin="hydrag",
                trust_level="local",
            ),
        ]

        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            with patch("hydrag_benchmark.heads.head_hydrag.hydrag_search", return_value=mock_results):
                results = head.retrieve("some query", n_results=5)

        assert len(results) == 0

    def test_retrieve_empty_results(self, sample_chunks: list[Chunk]) -> None:
        """Empty hydrag_search output → empty ScoredChunk list."""
        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            with patch("hydrag_benchmark.heads.head_hydrag.hydrag_search", return_value=[]):
                results = head.retrieve("nonexistent topic", n_results=5)

        assert results == []

    def test_retrieve_preserves_ranking_order(self, sample_chunks: list[Chunk]) -> None:
        """Output order matches hydrag_search result order."""
        from hydrag.fusion import RetrievalResult

        mock_results = [
            RetrievalResult(text=sample_chunks[3].text, source="", score=0.95, head_origin="hydrag", trust_level="local"),
            RetrievalResult(text=sample_chunks[1].text, source="", score=0.80, head_origin="hydrag", trust_level="local"),
            RetrievalResult(text=sample_chunks[4].text, source="", score=0.60, head_origin="hydrag", trust_level="local"),
        ]

        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            with patch("hydrag_benchmark.heads.head_hydrag.hydrag_search", return_value=mock_results):
                results = head.retrieve("reinforcement learning", n_results=3)

        assert [r.chunk.chunk_id for r in results] == ["doc-4", "doc-2", "doc-5"]
        assert results[0].score > results[1].score > results[2].score

    def test_retrieve_deduplicates_by_chunk_id(self, sample_chunks: list[Chunk]) -> None:
        """If hydrag_search returns the same text twice, both map to valid ScoredChunks
        (dedup is the caller's responsibility, not HeadHydrag's)."""
        from hydrag.fusion import RetrievalResult

        mock_results = [
            RetrievalResult(text=sample_chunks[0].text, source="", score=0.9, head_origin="hydrag", trust_level="local"),
            RetrievalResult(text=sample_chunks[0].text, source="", score=0.5, head_origin="hydrag", trust_level="local"),
        ]

        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            with patch("hydrag_benchmark.heads.head_hydrag.hydrag_search", return_value=mock_results):
                results = head.retrieve("machine learning", n_results=5)

        # Both resolve — both valid
        assert len(results) == 2
        assert all(r.chunk.chunk_id == "doc-1" for r in results)


# ── Config Tests ─────────────────────────────────────────────────────────────


class TestHeadHydragConfig:
    def test_web_fallback_disabled(self) -> None:
        """Web fallback must be off for offline benchmarking."""
        head = HeadHydrag()
        assert head._config.enable_head_3b_web is False
        assert head._config.enable_web_fallback is False
        head.close()

    def test_crag_enabled_by_default(self) -> None:
        head = HeadHydrag()
        assert head._config.enable_head_2_crag is True
        head.close()

    def test_crag_can_be_disabled(self) -> None:
        head = HeadHydrag(enable_crag=False)
        assert head._config.enable_head_2_crag is False
        head.close()

    def test_profile_prose_default(self) -> None:
        head = HeadHydrag()
        assert head._config.profile == "prose"
        head.close()

    def test_profile_code(self) -> None:
        head = HeadHydrag(profile="code")
        assert head._config.profile == "code"
        head.close()

    def test_custom_llm_passed_through(self, sample_chunks: list[Chunk]) -> None:
        """Custom LLM provider is forwarded to hydrag_search."""
        mock_llm = MagicMock()

        with HeadHydrag(llm=mock_llm) as head:
            head.build_index(sample_chunks)
            with patch("hydrag_benchmark.heads.head_hydrag.hydrag_search", return_value=[]) as mock_hs:
                head.retrieve("test query", n_results=5)
                call_kwargs = mock_hs.call_args
                assert call_kwargs.kwargs.get("llm") is mock_llm or call_kwargs[1].get("llm") is mock_llm


# ── Protocol Compliance ──────────────────────────────────────────────────────


class TestHeadHydragProtocol:
    def test_has_name_property(self) -> None:
        head = HeadHydrag()
        assert isinstance(head.name, str)
        head.close()

    def test_has_retrieve_method(self) -> None:
        head = HeadHydrag()
        assert callable(getattr(head, "retrieve", None))
        head.close()

    def test_retrieve_returns_scored_chunks(self, sample_chunks: list[Chunk]) -> None:
        from hydrag.fusion import RetrievalResult

        mock_results = [
            RetrievalResult(text=sample_chunks[0].text, source="", score=0.9, head_origin="hydrag", trust_level="local"),
        ]

        with HeadHydrag() as head:
            head.build_index(sample_chunks)
            with patch("hydrag_benchmark.heads.head_hydrag.hydrag_search", return_value=mock_results):
                results = head.retrieve("test", n_results=1)

        assert all(isinstance(r, ScoredChunk) for r in results)
