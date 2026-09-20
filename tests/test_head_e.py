"""Tests for HeadE — SQLite FTS5 + LLM enrichment retrieval (T-5052 A08/B-01).

The only prior coverage of HeadE (in test_beir.py) mocks the class
entirely, so it never exercised real build_index/retrieve behavior.
HeadE overrides HeadD.build_index() but never populated
self._text_to_id, so every _fts_search result's raw_content failed the
self._text_to_id.get(text) lookup in _fts_search_chunk_ids() and was
silently dropped -- every real HeadE query returned zero results.
"""

from __future__ import annotations

from unittest import mock

from hydrag_benchmark.heads.base import Chunk
from hydrag_benchmark.heads.head_e import HeadE


def _run_head_e(chunks: list[Chunk], query: str, n_results: int = 10):
    with mock.patch(
        "hydrag.enrichment.OllamaKeywordExtractor.extract",
        return_value={"summary": "", "keywords": []},
    ):
        with HeadE(db_path=":memory:") as head:
            head.build_index(chunks)
            return head.retrieve(query, n_results=n_results)


class TestHeadEBuildIndexAndRetrieve:
    def test_real_head_returns_non_zero_results(self) -> None:
        chunks = [
            Chunk(chunk_id="c1", text="Python uses SQLite for local storage", source="doc1"),
            Chunk(chunk_id="c2", text="Rust guarantees memory safety at compile time", source="doc2"),
            Chunk(chunk_id="c3", text="Kubernetes orchestrates containerized workloads", source="doc3"),
        ]
        results = _run_head_e(chunks, "SQLite storage")

        assert len(results) > 0, (
            "HeadE.build_index must populate the text-to-id map its own "
            "search path (_fts_search_chunk_ids) relies on"
        )
        assert results[0].chunk.chunk_id == "c1"
        assert all(r.head_origin == "fts5_enriched" for r in results)

    def test_build_index_populates_text_to_id_map(self) -> None:
        chunks = [Chunk(chunk_id="c1", text="unique searchable content", source="doc1")]
        with mock.patch(
            "hydrag.enrichment.OllamaKeywordExtractor.extract",
            return_value={"summary": "", "keywords": []},
        ):
            with HeadE(db_path=":memory:") as head:
                head.build_index(chunks)
                assert head._text_to_id == {"unique searchable content": "c1"}

    def test_multiple_queries_all_find_their_chunk(self) -> None:
        chunks = [
            Chunk(chunk_id=f"c{i}", text=f"document number {i} about topic {i}", source=f"doc{i}")
            for i in range(5)
        ]
        for i in range(5):
            results = _run_head_e(chunks, f"topic {i}")
            chunk_ids = {r.chunk.chunk_id for r in results}
            assert f"c{i}" in chunk_ids, f"query for topic {i} found no results at all"
