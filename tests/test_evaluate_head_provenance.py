"""T-5052 A11 (B-05): _evaluate_head must surface per-query head_origin
counts and fast-path/crag-skipped counts, so a run in which every
result actually came from Head 0's BM25 fast path is visible in the
result JSON rather than indistinguishable from a full-pipeline run.
"""

from __future__ import annotations

from hydrag_benchmark.beir_runner import _evaluate_head
from hydrag_benchmark.heads.base import Chunk, ScoredChunk


class _FakeHead:
    """A RetrievalHead stub returning pre-scripted ScoredChunks per query."""

    name = "hydrag_full"

    def __init__(self, scripted: dict[str, list[ScoredChunk]]) -> None:
        self._scripted = scripted

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        return self._scripted.get(query, [])


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, text=f"text {cid}", source="doc")


class TestEvaluateHeadProvenance:
    def test_all_fast_path_query_is_visible(self) -> None:
        scripted = {
            "some query": [
                ScoredChunk(
                    chunk=_chunk("d1"), score=1.0, head_origin="hydrag_full",
                    metadata={"hydrag_head_origin": "head_0", "fast_path_triggered": True, "crag_skipped": True},
                ),
                ScoredChunk(
                    chunk=_chunk("d2"), score=0.9, head_origin="hydrag_full",
                    metadata={"hydrag_head_origin": "head_0", "fast_path_triggered": True, "crag_skipped": True},
                ),
            ],
        }
        head = _FakeHead(scripted)
        result = _evaluate_head(
            head,
            queries={"q1": "some query"},
            qrels={"q1": {"d1": 1}},
            chunk_to_doc={"d1": "d1", "d2": "d2"},
            dataset_name="test",
            n_corpus=2,
        )
        qr = result.queries[0]
        assert qr.head_origin_counts == {"head_0": 2}
        assert qr.fast_path_count == 2
        assert qr.crag_skipped_count == 2
        # Aggregate on HeadResult must reflect it too, not just per-query.
        assert result.head_origin_counts == {"head_0": 2}
        assert result.fast_path_count == 2
        assert result.crag_skipped_count == 2

    def test_mixed_origins_are_tallied_separately(self) -> None:
        scripted = {
            "some query": [
                ScoredChunk(
                    chunk=_chunk("d1"), score=1.0, head_origin="hydrag_full",
                    metadata={"hydrag_head_origin": "head_0", "fast_path_triggered": True, "crag_skipped": False},
                ),
                ScoredChunk(
                    chunk=_chunk("d2"), score=0.5, head_origin="hydrag_full",
                    metadata={"hydrag_head_origin": "head_3a", "fast_path_triggered": False, "crag_skipped": False},
                ),
            ],
        }
        head = _FakeHead(scripted)
        result = _evaluate_head(
            head,
            queries={"q1": "some query"},
            qrels={"q1": {"d1": 1, "d2": 1}},
            chunk_to_doc={"d1": "d1", "d2": "d2"},
            dataset_name="test",
            n_corpus=2,
        )
        qr = result.queries[0]
        assert qr.head_origin_counts == {"head_0": 1, "head_3a": 1}
        assert qr.fast_path_count == 1
        assert qr.crag_skipped_count == 0

    def test_heads_without_provenance_metadata_report_no_counts(self) -> None:
        """Head D/E don't populate ScoredChunk.metadata -- must not crash
        and must report empty/zero counts, not fabricate data."""
        scripted = {
            "some query": [ScoredChunk(chunk=_chunk("d1"), score=1.0, head_origin="fts5_baseline")],
        }
        head = _FakeHead(scripted)
        result = _evaluate_head(
            head,
            queries={"q1": "some query"},
            qrels={"q1": {"d1": 1}},
            chunk_to_doc={"d1": "d1"},
            dataset_name="test",
            n_corpus=1,
        )
        qr = result.queries[0]
        assert qr.head_origin_counts == {}
        assert qr.fast_path_count == 0
        assert qr.crag_skipped_count == 0
