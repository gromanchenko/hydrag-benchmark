"""Tests for strategy dispatch — validates all SUPPORTED_STRATEGIES."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydrag_benchmark.cli import main
from hydrag_benchmark.runner import SUPPORTED_STRATEGIES, _ChromaDBAdapter


SUITE_YAML = """\
name: strategy-test-suite
version: "1.0"
seed: 42
description: Suite for strategy dispatch tests.

environment:
  strategy: similarity
  n_results: 2

cases:
  - id: strat-001
    query: "fibonacci function"
    relevant_phrases:
      - "fibonacci"
    tags: [strategy]
  - id: strat-002
    query: "parse yaml"
    relevant_phrases:
      - "yaml"
      - "parse"
    tags: [strategy]
"""

CORPUS_FILE = """\
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def parse_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
"""


@pytest.fixture()
def bench_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    suite = tmp_path / "suite.yaml"
    suite.write_text(SUITE_YAML)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "sample.py").write_text(CORPUS_FILE)
    output = tmp_path / "output"
    return suite, corpus, output


class TestSupportedStrategies:
    def test_supported_strategies_constant(self) -> None:
        assert "similarity" in SUPPORTED_STRATEGIES
        assert "hybrid" in SUPPORTED_STRATEGIES
        assert "crag" in SUPPORTED_STRATEGIES
        assert "hydrag" in SUPPORTED_STRATEGIES

    def test_unknown_strategy_raises(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, _ = bench_env
        with pytest.raises(ValueError, match="Unknown strategy"):
            main([
                "run", str(suite),
                "--strategy", "nonexistent",
                "--corpus-dir", str(corpus),
            ])


class TestSimilarityStrategy:
    """Baseline strategy — direct chromadb query, no hydrag-core pipeline."""

    def test_similarity_produces_results(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, _ = bench_env
        rc = main([
            "run", str(suite),
            "--strategy", "similarity",
            "--corpus-dir", str(corpus),
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["strategy"] == "similarity"
        assert len(data["cases"]) == 2

    def test_similarity_recall(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, _ = bench_env
        main([
            "run", str(suite),
            "--strategy", "similarity",
            "--corpus-dir", str(corpus),
        ])
        data = json.loads(capsys.readouterr().out)
        assert data["summary"]["recall_at_k"] > 0.0


class TestHybridStrategy:
    """Hybrid — hydrag_search with head_1 only."""

    def test_hybrid_produces_results(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, _ = bench_env
        rc = main([
            "run", str(suite),
            "--strategy", "hybrid",
            "--corpus-dir", str(corpus),
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["strategy"] == "hybrid"
        assert len(data["cases"]) == 2


class TestCragStrategy:
    """CRAG — heads 1 + 2 + 3a. CRAG falls back gracefully without LLM."""

    def test_crag_produces_results(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, _ = bench_env
        rc = main([
            "run", str(suite),
            "--strategy", "crag",
            "--corpus-dir", str(corpus),
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["strategy"] == "crag"
        assert len(data["cases"]) == 2


class TestHydragStrategy:
    """Full hydrag pipeline — all default heads."""

    def test_hydrag_produces_results(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, _ = bench_env
        rc = main([
            "run", str(suite),
            "--strategy", "hydrag",
            "--corpus-dir", str(corpus),
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["strategy"] == "hydrag"
        assert len(data["cases"]) == 2


class TestStrategyDeterminism:
    """Same strategy + seed → identical scores across runs."""

    @pytest.mark.parametrize("strategy", sorted(SUPPORTED_STRATEGIES))
    def test_seeded_determinism(
        self,
        strategy: str,
        bench_env: tuple[Path, Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        suite, corpus, _ = bench_env
        main([
            "run", str(suite),
            "--strategy", strategy,
            "--corpus-dir", str(corpus),
            "--seed", "42",
        ])
        out1 = json.loads(capsys.readouterr().out)

        main([
            "run", str(suite),
            "--strategy", strategy,
            "--corpus-dir", str(corpus),
            "--seed", "42",
        ])
        out2 = json.loads(capsys.readouterr().out)

        for c1, c2 in zip(out1["cases"], out2["cases"]):
            assert c1["recall_at_1"] == c2["recall_at_1"]
            assert c1["recall_at_k"] == c2["recall_at_k"]
            assert c1["mrr"] == c2["mrr"]


class TestChromaDBAdapter:
    """Unit tests for the _ChromaDBAdapter protocol implementation."""

    def test_adapter_protocol_compliance(self) -> None:
        from hydrag import VectorStoreAdapter

        class FakeCollection:
            def query(self, query_texts: list[str], n_results: int = 5) -> dict:
                return {"documents": [["doc1", "doc2"]]}

        adapter = _ChromaDBAdapter(FakeCollection(), n_results=2)
        assert isinstance(adapter, VectorStoreAdapter)

    def test_adapter_delegates_to_collection(self) -> None:
        calls: list[tuple[list[str], int]] = []

        class SpyCollection:
            def query(self, query_texts: list[str], n_results: int = 5) -> dict:
                calls.append((query_texts, n_results))
                return {"documents": [["result1"]]}

        adapter = _ChromaDBAdapter(SpyCollection(), n_results=3)
        result = adapter.semantic_search("test query", n_results=3)
        assert result == ["result1"]
        assert len(calls) == 1
        assert calls[0] == (["test query"], 3)

        result2 = adapter.keyword_search("another", n_results=5)
        assert result2 == ["result1"]

        result3 = adapter.hybrid_search("third", n_results=2)
        assert result3 == ["result1"]
        assert len(calls) == 3
