"""Tests for Head D (FTS5 raw), Head E (FTS5 + enrichment), BEIR loader, and BEIR metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from hydrag_benchmark.beir_loader import load_beir_corpus, load_beir_qrels, load_beir_queries
from hydrag_benchmark.beir_runner import map_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from hydrag_benchmark.heads.base import Chunk, ScoredChunk
from hydrag_benchmark.heads.head_d import HeadD


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


@pytest.fixture()
def beir_dataset(tmp_path: Path) -> Path:
    """Create a minimal BEIR-format dataset on disk."""
    ds_dir = tmp_path / "test-dataset"
    ds_dir.mkdir()

    # corpus.jsonl
    corpus = [
        {"_id": "1", "title": "ML Overview", "text": "Machine learning transforms data."},
        {"_id": "2", "title": "Deep Learning", "text": "Neural networks for vision and language."},
        {"_id": "3", "title": "NLP Methods", "text": "Text classification and entity recognition."},
    ]
    with open(ds_dir / "corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    # queries.jsonl
    queries = [
        {"_id": "q1", "text": "What is machine learning?"},
        {"_id": "q2", "text": "How do neural networks work?"},
    ]
    with open(ds_dir / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    # qrels/test.tsv
    (ds_dir / "qrels").mkdir()
    with open(ds_dir / "qrels" / "test.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("q1\t1\t1\n")
        f.write("q2\t2\t1\n")

    return ds_dir


# ── Head D Tests ─────────────────────────────────────────────────────────────


class TestHeadD:
    def test_name(self) -> None:
        head = HeadD()
        assert head.name == "head_d"
        head.close()

    def test_build_index_and_retrieve(self, sample_chunks: list[Chunk]) -> None:
        head = HeadD()
        head.build_index(sample_chunks)
        results = head.retrieve("machine learning algorithms", n_results=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredChunk) for r in results)
        assert results[0].head_origin == "head_d"
        head.close()

    def test_scores_descending(self, sample_chunks: list[Chunk]) -> None:
        head = HeadD()
        head.build_index(sample_chunks)
        results = head.retrieve("learning", n_results=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        head.close()

    def test_empty_query(self, sample_chunks: list[Chunk]) -> None:
        head = HeadD()
        head.build_index(sample_chunks)
        results = head.retrieve("", n_results=5)
        assert results == []
        head.close()

    def test_no_match(self, sample_chunks: list[Chunk]) -> None:
        head = HeadD()
        head.build_index(sample_chunks)
        results = head.retrieve("xyzzyplugh", n_results=5)
        assert results == []
        head.close()

    def test_context_manager(self, sample_chunks: list[Chunk]) -> None:
        with HeadD() as head:
            head.build_index(sample_chunks)
            results = head.retrieve("neural", n_results=3)
            assert len(results) > 0

    def test_retrieve_correct_chunk(self, sample_chunks: list[Chunk]) -> None:
        head = HeadD()
        head.build_index(sample_chunks)
        results = head.retrieve("support vector machines classification", n_results=1)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "doc-5"
        head.close()


# ── BEIR Loader Tests ────────────────────────────────────────────────────────


class TestBeirLoader:
    def test_load_corpus(self, beir_dataset: Path) -> None:
        corpus = load_beir_corpus(beir_dataset)
        assert len(corpus) == 3
        assert "1" in corpus
        assert corpus["1"]["title"] == "ML Overview"
        assert "transforms" in corpus["1"]["text"]

    def test_load_queries(self, beir_dataset: Path) -> None:
        queries = load_beir_queries(beir_dataset)
        assert len(queries) == 2
        assert "q1" in queries
        assert "machine learning" in queries["q1"]

    def test_load_qrels(self, beir_dataset: Path) -> None:
        qrels = load_beir_qrels(beir_dataset)
        assert len(qrels) == 2
        assert "q1" in qrels
        assert qrels["q1"]["1"] == 1
        assert "q2" in qrels
        assert qrels["q2"]["2"] == 1

    def test_load_qrels_filters_zero_relevance(self, beir_dataset: Path) -> None:
        # Add a zero-relevance entry
        qrel_path = beir_dataset / "qrels" / "test.tsv"
        with open(qrel_path, "a") as f:
            f.write("q1\t3\t0\n")
        qrels = load_beir_qrels(beir_dataset)
        assert "3" not in qrels.get("q1", {})


# ── BEIR Metrics Tests ───────────────────────────────────────────────────────


class TestBeirMetrics:
    def test_ndcg_perfect(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        qrel = {"d1": 1, "d2": 1, "d3": 1}
        assert ndcg_at_k(retrieved, qrel, k=3) == pytest.approx(1.0)

    def test_ndcg_no_match(self) -> None:
        retrieved = ["d4", "d5", "d6"]
        qrel = {"d1": 1, "d2": 1}
        assert ndcg_at_k(retrieved, qrel, k=3) == 0.0

    def test_ndcg_partial(self) -> None:
        retrieved = ["d2", "d1"]
        qrel = {"d1": 2, "d2": 1}
        score = ndcg_at_k(retrieved, qrel, k=2)
        assert 0.0 < score < 1.0

    def test_recall_perfect(self) -> None:
        retrieved = ["d1", "d2"]
        qrel = {"d1": 1, "d2": 1}
        assert recall_at_k(retrieved, qrel, k=2) == 1.0

    def test_recall_partial(self) -> None:
        retrieved = ["d1", "d3"]
        qrel = {"d1": 1, "d2": 1}
        assert recall_at_k(retrieved, qrel, k=2) == 0.5

    def test_recall_empty_qrel(self) -> None:
        assert recall_at_k(["d1"], {}, k=1) == 0.0

    def test_mrr_first(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        qrel = {"d1": 1}
        assert mrr_at_k(retrieved, qrel, k=3) == 1.0

    def test_mrr_second(self) -> None:
        retrieved = ["d2", "d1"]
        qrel = {"d1": 1}
        assert mrr_at_k(retrieved, qrel, k=2) == 0.5

    def test_mrr_no_match(self) -> None:
        retrieved = ["d3", "d4"]
        qrel = {"d1": 1}
        assert mrr_at_k(retrieved, qrel, k=2) == 0.0

    def test_map_perfect_order(self) -> None:
        retrieved = ["d1", "d2"]
        qrel = {"d1": 1, "d2": 1}
        assert map_at_k(retrieved, qrel, k=2) == 1.0

    def test_map_second_only(self) -> None:
        retrieved = ["d3", "d1"]
        qrel = {"d1": 1}
        # precision@2 = 1/2, only 1 relevant doc → AP = 0.5/1 = 0.5
        assert map_at_k(retrieved, qrel, k=2) == pytest.approx(0.5)


# ── Integration: Head D + BEIR dataset ───────────────────────────────────────


class TestHeadDBeirIntegration:
    def test_end_to_end(self, beir_dataset: Path) -> None:
        """Index a BEIR dataset into Head D and evaluate."""
        corpus = load_beir_corpus(beir_dataset)
        queries = load_beir_queries(beir_dataset)
        qrels = load_beir_qrels(beir_dataset)

        # Build chunks
        chunks: list[Chunk] = []
        chunk_to_doc: dict[str, str] = {}
        for doc_id, doc in corpus.items():
            text = f"{doc['title']}\n{doc['text']}".strip()
            cid = f"doc-{doc_id}"
            chunks.append(Chunk(chunk_id=cid, text=text, source=f"beir:{doc_id}"))
            chunk_to_doc[cid] = doc_id

        head = HeadD()
        head.build_index(chunks)

        # Run query and compute metrics
        for qid, qtext in queries.items():
            if qid not in qrels:
                continue
            results = head.retrieve(qtext, n_results=3)
            doc_ids = []
            for sc in results:
                did = chunk_to_doc.get(sc.chunk.chunk_id, "")
                if did:
                    doc_ids.append(did)

            ndcg = ndcg_at_k(doc_ids, qrels[qid], k=3)
            recall = recall_at_k(doc_ids, qrels[qid], k=3)
            # Metrics should be computable (no crashes)
            assert 0.0 <= ndcg <= 1.0
            assert 0.0 <= recall <= 1.0

        head.close()
