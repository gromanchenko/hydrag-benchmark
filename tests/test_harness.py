"""Tests for the multi-head combination harness and Head C."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydrag_benchmark.embedding import HashEmbedder
from hydrag_benchmark.harness import CONFIGS, _merge_results, run_multihead_benchmark
from hydrag_benchmark.heads.base import Chunk, ScoredChunk
from hydrag_benchmark.heads.head_a import HeadA
from hydrag_benchmark.heads.head_b import HeadB, HeadBIndex, VectorEntry
from hydrag_benchmark.heads.head_c import HeadC
from hydrag_benchmark.suite import BenchSuite


# ── Helpers ──────────────────────────────────────────────────────────────────

SUITE_YAML = """\
name: multihead-test
version: "1.0"
seed: 42
description: Suite for multi-head harness tests.

environment:
  strategy: multihead
  n_results: 3

cases:
  - id: mh-001
    query: "fibonacci recursive function"
    relevant_phrases:
      - "fibonacci"
      - "recursive"
    tags: [multihead]
  - id: mh-002
    query: "read csv file"
    relevant_phrases:
      - "csv"
      - "read"
    tags: [multihead]
"""


class _FakeDoc2Query:
    def generate(self, chunk_text: str) -> list[str]:
        return [
            "How does the main function in this code chunk handle its input processing?",
            "Why does this implementation fail when provided with empty input data?",
            "What are the performance implications of this particular code approach?",
        ]


@pytest.fixture()
def test_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="ch-1", source="math.py",
            text="def fibonacci(n):\n    \"\"\"Recursive fibonacci.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        ),
        Chunk(
            chunk_id="ch-2", source="io.py",
            text="import csv\n\ndef read_csv(path):\n    with open(path) as f:\n        reader = csv.reader(f)\n        return list(reader)",
        ),
        Chunk(
            chunk_id="ch-3", source="main.py",
            text="from math import fibonacci\nfrom io import read_csv\n\nresult = fibonacci(10)\ndata = read_csv('input.csv')",
        ),
    ]


@pytest.fixture()
def built_heads(test_chunks: list[Chunk]) -> tuple[HeadA, HeadB, HeadC]:
    embedder = HashEmbedder(dimension=64)
    head_a = HeadA(test_chunks)
    head_b = HeadB(embedder=embedder, doc2query=_FakeDoc2Query())
    head_b.build_index(test_chunks)
    head_c = HeadC(head_a=head_a, head_b=head_b)
    return head_a, head_b, head_c


@pytest.fixture()
def test_suite(tmp_path: Path) -> BenchSuite:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(SUITE_YAML)
    return BenchSuite.from_yaml(suite_path)


# ── Head C tests ─────────────────────────────────────────────────────────────


class TestHeadC:
    def test_name(self, built_heads: tuple[HeadA, HeadB, HeadC]) -> None:
        _, _, head_c = built_heads
        assert head_c.name == "head_c"

    def test_retrieval_returns_results(self, built_heads: tuple[HeadA, HeadB, HeadC]) -> None:
        _, _, head_c = built_heads
        results = head_c.retrieve("fibonacci recursive", n_results=5)
        assert len(results) > 0

    def test_retrieval_scores_ordered(self, built_heads: tuple[HeadA, HeadB, HeadC]) -> None:
        _, _, head_c = built_heads
        results = head_c.retrieve("fibonacci", n_results=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_head_origin(self, built_heads: tuple[HeadA, HeadB, HeadC]) -> None:
        _, _, head_c = built_heads
        results = head_c.retrieve("fibonacci", n_results=5)
        for r in results:
            assert r.head_origin == "head_c"

    def test_no_candidates_returns_empty(
        self, built_heads: tuple[HeadA, HeadB, HeadC]
    ) -> None:
        _, _, head_c = built_heads
        results = head_c.retrieve("completely unrelated topic about cooking")
        assert len(results) == 0


# ── Merge results ────────────────────────────────────────────────────────────


class TestMergeResults:
    def test_deduplication(self, test_chunks: list[Chunk]) -> None:
        a = [ScoredChunk(chunk=test_chunks[0], score=0.8, head_origin="a")]
        b = [ScoredChunk(chunk=test_chunks[0], score=0.9, head_origin="b")]
        merged = _merge_results(a, b, n_results=10)
        assert len(merged) == 1
        assert merged[0].score == 0.9  # Keep higher score

    def test_union(self, test_chunks: list[Chunk]) -> None:
        a = [ScoredChunk(chunk=test_chunks[0], score=0.8, head_origin="a")]
        b = [ScoredChunk(chunk=test_chunks[1], score=0.7, head_origin="b")]
        merged = _merge_results(a, b, n_results=10)
        assert len(merged) == 2

    def test_n_results_limit(self, test_chunks: list[Chunk]) -> None:
        items = [
            ScoredChunk(chunk=c, score=0.5, head_origin="a") for c in test_chunks
        ]
        merged = _merge_results(items, n_results=2)
        assert len(merged) == 2

    def test_sorted_by_score(self, test_chunks: list[Chunk]) -> None:
        a = [ScoredChunk(chunk=test_chunks[0], score=0.3, head_origin="a")]
        b = [ScoredChunk(chunk=test_chunks[1], score=0.9, head_origin="b")]
        c = [ScoredChunk(chunk=test_chunks[2], score=0.6, head_origin="c")]
        merged = _merge_results(a, b, c, n_results=10)
        scores = [m.score for m in merged]
        assert scores == sorted(scores, reverse=True)


# ── Harness configs ──────────────────────────────────────────────────────────


class TestConfigs:
    def test_all_configs_defined(self) -> None:
        assert CONFIGS == ["A-only", "B-only", "C-only", "A+B", "A+B+C"]


# ── Full harness integration ─────────────────────────────────────────────────


class TestHarness:
    def test_full_run(
        self,
        built_heads: tuple[HeadA, HeadB, HeadC],
        test_chunks: list[Chunk],
        test_suite: BenchSuite,
    ) -> None:
        head_a, head_b, head_c = built_heads
        result_json = run_multihead_benchmark(
            suite=test_suite,
            chunks=test_chunks,
            head_a=head_a,
            head_b=head_b,
            head_c=head_c,
            corpus_dir="/tmp/test-corpus",
            n_results=3,
        )
        data = json.loads(result_json)
        assert data["schema_version"] == "0.2"
        assert data["suite_name"] == "multihead-test"
        assert len(data["configs"]) == 5

    def test_all_configs_have_cases(
        self,
        built_heads: tuple[HeadA, HeadB, HeadC],
        test_chunks: list[Chunk],
        test_suite: BenchSuite,
    ) -> None:
        head_a, head_b, head_c = built_heads
        result_json = run_multihead_benchmark(
            suite=test_suite,
            chunks=test_chunks,
            head_a=head_a,
            head_b=head_b,
            head_c=head_c,
            corpus_dir="/tmp/test",
            n_results=3,
        )
        data = json.loads(result_json)
        for cfg in data["configs"]:
            assert len(cfg["cases"]) == 2
            assert cfg["summary"]["total_cases"] == 2

    def test_metrics_present(
        self,
        built_heads: tuple[HeadA, HeadB, HeadC],
        test_chunks: list[Chunk],
        test_suite: BenchSuite,
    ) -> None:
        head_a, head_b, head_c = built_heads
        result_json = run_multihead_benchmark(
            suite=test_suite,
            chunks=test_chunks,
            head_a=head_a,
            head_b=head_b,
            head_c=head_c,
            corpus_dir="/tmp/test",
        )
        data = json.loads(result_json)
        for cfg in data["configs"]:
            summary = cfg["summary"]
            assert "recall_at_1" in summary
            assert "recall_at_k" in summary
            assert "mrr" in summary
            assert "chunk_overlap" in summary
            assert "latency_ms" in summary

    def test_filter_stats_in_output(
        self,
        built_heads: tuple[HeadA, HeadB, HeadC],
        test_chunks: list[Chunk],
        test_suite: BenchSuite,
    ) -> None:
        head_a, head_b, head_c = built_heads
        result_json = run_multihead_benchmark(
            suite=test_suite,
            chunks=test_chunks,
            head_a=head_a,
            head_b=head_b,
            head_c=head_c,
            corpus_dir="/tmp/test",
        )
        data = json.loads(result_json)
        assert "head_b_filter_stats" in data
        assert "head_b_rejection_rate" in data

    def test_sidecar_written(
        self,
        built_heads: tuple[HeadA, HeadB, HeadC],
        test_chunks: list[Chunk],
        test_suite: BenchSuite,
        tmp_path: Path,
    ) -> None:
        head_a, head_b, head_c = built_heads
        sidecar = tmp_path / "sidecar.json"
        run_multihead_benchmark(
            suite=test_suite,
            chunks=test_chunks,
            head_a=head_a,
            head_b=head_b,
            head_c=head_c,
            corpus_dir="/tmp/test",
            sidecar_path=sidecar,
        )
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert isinstance(data, dict)
