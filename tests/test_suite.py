"""Tests for hydrag_benchmark.suite — YAML loader."""

from pathlib import Path

from hydrag_benchmark.suite import BenchSuite


SUITE_YAML = """\
name: test-suite
version: "1.0"
seed: 42
description: Test suite for unit tests.

environment:
  strategy: hybrid
  n_results: 3

cases:
  - id: t-001
    query: "fibonacci function"
    relevant_phrases:
      - "fibonacci"
      - "def fib"
    tags: [test]
  - id: t-002
    query: "yaml parser"
    relevant_phrases:
      - "yaml"
    tags: [test, config]
"""


class TestBenchSuite:
    def test_from_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "suite.yaml"
        p.write_text(SUITE_YAML)
        suite = BenchSuite.from_yaml(p)
        assert suite.name == "test-suite"
        assert suite.version == "1.0"
        assert suite.strategy == "hybrid"
        assert suite.n_results == 3
        assert suite.seed == 42
        assert len(suite.cases) == 2
        assert suite.cases[0].id == "t-001"
        assert suite.cases[0].relevant_phrases == ["fibonacci", "def fib"]
        assert suite.cases[1].tags == ["test", "config"]

    def test_strategy_override(self, tmp_path: Path) -> None:
        p = tmp_path / "suite.yaml"
        p.write_text(SUITE_YAML)
        suite = BenchSuite.from_yaml(p, strategy_override="hydrag")
        assert suite.strategy == "hydrag"

    def test_n_results_override(self, tmp_path: Path) -> None:
        p = tmp_path / "suite.yaml"
        p.write_text(SUITE_YAML)
        suite = BenchSuite.from_yaml(p, n_results_override=10)
        assert suite.n_results == 10

    def test_seed_override(self, tmp_path: Path) -> None:
        p = tmp_path / "suite.yaml"
        p.write_text(SUITE_YAML)
        suite = BenchSuite.from_yaml(p, seed_override=99)
        assert suite.seed == 99
