"""Tests for CLI — smoke test and schema conformance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hydrag_benchmark.cli import main


SUITE_YAML = """\
name: cli-test-suite
version: "1.0"
seed: 42
description: Minimal suite for CLI tests.

environment:
  strategy: hydrag
  n_results: 2

cases:
  - id: cli-001
    query: "fibonacci function"
    relevant_phrases:
      - "fibonacci"
    tags: [cli-test]
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


REQUIRED_SCHEMA_KEYS = {
    "schema_version",
    "run_id",
    "timestamp",
    "tool_version",
    "suite_name",
    "strategy",
    "n_results",
    "seed",
    "corpus_dir",
    "cases",
    "summary",
}

REQUIRED_CASE_KEYS = {
    "case_id",
    "query",
    "strategy",
    "recall_at_1",
    "recall_at_k",
    "mrr",
    "chunk_overlap",
    "latency_ms",
}

REQUIRED_SUMMARY_KEYS = {
    "total_cases",
    "recall_at_1",
    "recall_at_k",
    "mrr",
    "chunk_overlap",
    "latency_ms",
}

REQUIRED_LATENCY_KEYS = {"avg", "p50", "p95", "p99"}


@pytest.fixture()
def bench_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create suite YAML, corpus dir, and output dir."""
    suite = tmp_path / "suite.yaml"
    suite.write_text(SUITE_YAML)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "sample.py").write_text(CORPUS_FILE)
    output = tmp_path / "output"
    return suite, corpus, output


class TestCLIHelp:
    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([])
        assert rc == 0
        captured = capsys.readouterr()
        assert "hydrag-bench" in captured.out

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0

    def test_list_suites(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (tmp_path / "a.yaml").write_text("name: a\ncases: []\n")
        (tmp_path / "b.yaml").write_text("name: b\ncases: []\n")
        rc = main(["list-suites", "--suite-dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "a.yaml" in out
        assert "b.yaml" in out


class TestCLIRun:
    def test_run_produces_valid_json(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, output = bench_env
        rc = main([
            "run", str(suite),
            "--strategy", "hydrag",
            "--corpus-dir", str(corpus),
            "--n-results", "2",
            "--seed", "42",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["schema_version"] == "0.1"

    def test_schema_conformance(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite, corpus, output = bench_env
        main([
            "run", str(suite),
            "--strategy", "hydrag",
            "--corpus-dir", str(corpus),
        ])
        data = json.loads(capsys.readouterr().out)

        # Top-level keys
        assert REQUIRED_SCHEMA_KEYS.issubset(set(data.keys()))
        assert data["schema_version"] == "0.1"
        assert data["suite_name"] == "cli-test-suite"
        assert data["strategy"] == "hydrag"

        # Case keys
        assert len(data["cases"]) >= 1
        for case in data["cases"]:
            assert REQUIRED_CASE_KEYS.issubset(set(case.keys()))

        # Summary keys
        assert REQUIRED_SUMMARY_KEYS.issubset(set(data["summary"].keys()))
        assert REQUIRED_LATENCY_KEYS.issubset(set(data["summary"]["latency_ms"].keys()))

    def test_output_dir(self, bench_env: tuple[Path, Path, Path]) -> None:
        suite, corpus, output = bench_env
        rc = main([
            "run", str(suite),
            "--strategy", "hydrag",
            "--corpus-dir", str(corpus),
            "--output-dir", str(output),
        ])
        assert rc == 0
        files = list(output.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["schema_version"] == "0.1"


class TestDeterminism:
    def test_seeded_runs_produce_identical_scores(
        self, bench_env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two runs with --seed 42 must produce identical metric values."""
        suite, corpus, _ = bench_env

        main([
            "run", str(suite),
            "--strategy", "hydrag",
            "--corpus-dir", str(corpus),
            "--seed", "42",
        ])
        out1 = json.loads(capsys.readouterr().out)

        main([
            "run", str(suite),
            "--strategy", "hydrag",
            "--corpus-dir", str(corpus),
            "--seed", "42",
        ])
        out2 = json.loads(capsys.readouterr().out)

        # Compare scores (ignore run_id, timestamp)
        for c1, c2 in zip(out1["cases"], out2["cases"]):
            assert c1["recall_at_1"] == c2["recall_at_1"]
            assert c1["recall_at_k"] == c2["recall_at_k"]
            assert c1["mrr"] == c2["mrr"]
            assert c1["chunk_overlap"] == c2["chunk_overlap"]

        assert out1["summary"]["recall_at_1"] == out2["summary"]["recall_at_1"]
        assert out1["summary"]["mrr"] == out2["summary"]["mrr"]
