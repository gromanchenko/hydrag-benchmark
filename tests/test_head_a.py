"""Tests for Head A — graph/index retrieval."""

from __future__ import annotations

import pytest

from hydrag_benchmark.heads.base import Chunk
from hydrag_benchmark.heads.head_a import (
    Graph,
    HeadA,
    SymbolIndex,
    extract_query_identifiers,
    extract_symbols,
)


# ── Symbol extraction ────────────────────────────────────────────────────────


class TestExtractSymbols:
    def test_python_functions(self) -> None:
        code = "def fibonacci(n):\n    return n\n\nclass Calculator:\n    pass"
        symbols = extract_symbols(code, ".py")
        assert "fibonacci" in symbols
        assert "Calculator" in symbols

    def test_python_imports(self) -> None:
        code = "import os\nfrom pathlib import Path\nimport json"
        symbols = extract_symbols(code, ".py")
        assert "os" in symbols
        assert "pathlib" in symbols

    def test_go_functions(self) -> None:
        code = "func handleRequest(w http.ResponseWriter, r *http.Request) {\n}"
        symbols = extract_symbols(code, ".go")
        assert "handleRequest" in symbols

    def test_js_declarations(self) -> None:
        code = "const fetchData = async () => {};\nfunction processItem(item) {}"
        symbols = extract_symbols(code, ".js")
        assert "fetchData" in symbols
        assert "processItem" in symbols

    def test_rust_items(self) -> None:
        code = "fn calculate_score(input: &str) -> f64 {\n    0.0\n}\nstruct Config {}"
        symbols = extract_symbols(code, ".rs")
        assert "calculate_score" in symbols
        assert "Config" in symbols

    def test_unknown_extension_uses_generic(self) -> None:
        code = "some_identifier another_thing"
        symbols = extract_symbols(code, ".xyz")
        assert "some_identifier" in symbols
        assert "another_thing" in symbols

    def test_deduplication(self) -> None:
        code = "def foo():\n    pass\ndef foo():\n    pass"
        symbols = extract_symbols(code, ".py")
        assert symbols.count("foo") == 1

    def test_filters_single_char(self) -> None:
        code = "x = 1\ny = 2\nlonger_name = 3"
        symbols = extract_symbols(code, ".py")
        assert "longer_name" in symbols


class TestExtractQueryIdentifiers:
    def test_basic_query(self) -> None:
        ids = extract_query_identifiers("function that calculates fibonacci numbers")
        assert "function" in ids
        assert "calculates" in ids
        assert "fibonacci" in ids

    def test_filters_stop_words(self) -> None:
        ids = extract_query_identifiers("the function is in the module")
        assert "the" not in ids
        assert "is" not in ids
        assert "in" not in ids
        assert "function" in ids
        assert "module" in ids

    def test_preserves_identifiers(self) -> None:
        ids = extract_query_identifiers("parse_yaml function in config module")
        assert "parse_yaml" in ids
        assert "config" in ids


# ── SymbolIndex ──────────────────────────────────────────────────────────────


class TestSymbolIndex:
    def test_add_and_lookup(self) -> None:
        idx = SymbolIndex()
        idx.add("fibonacci", "chunk-001")
        idx.add("fibonacci", "chunk-002")
        assert idx.lookup("fibonacci") == {"chunk-001", "chunk-002"}

    def test_case_insensitive(self) -> None:
        idx = SymbolIndex()
        idx.add("MyClass", "chunk-001")
        assert idx.lookup("myclass") == {"chunk-001"}
        assert idx.lookup("MYCLASS") == {"chunk-001"}

    def test_missing_symbol(self) -> None:
        idx = SymbolIndex()
        assert idx.lookup("nonexistent") == set()


# ── Graph ────────────────────────────────────────────────────────────────────


class TestGraph:
    def test_add_edge_and_neighbors(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert "b" in g.neighbors("a")
        assert "c" in g.neighbors("a")
        # Reverse
        assert "a" in g.neighbors("b")

    def test_in_degree(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("c", "b")
        g.add_edge("d", "b")
        assert g.in_degree("b") == 3
        assert g.in_degree("a") == 0

    def test_nodes(self) -> None:
        g = Graph()
        g.add_edge("x", "y")
        assert g.nodes() == {"x", "y"}


# ── HeadA integration ────────────────────────────────────────────────────────


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="ch-001", source="utils.py",
            text="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        ),
        Chunk(
            chunk_id="ch-002", source="parser.py",
            text="import yaml\n\ndef parse_yaml(path):\n    with open(path) as f:\n        return yaml.safe_load(f)",
        ),
        Chunk(
            chunk_id="ch-003", source="main.py",
            text="from utils import fibonacci\nfrom parser import parse_yaml\n\nresult = fibonacci(10)\nconfig = parse_yaml('config.yaml')",
        ),
        Chunk(
            chunk_id="ch-004", source="README.md",
            text="# Utils\n\nSee also `fibonacci` for number sequences.\nSee [parser docs](parser.md) for YAML parsing.",
        ),
    ]


class TestHeadA:
    def test_builds_symbol_index(self, sample_chunks: list[Chunk]) -> None:
        head = HeadA(sample_chunks)
        assert len(head.symbol_index) > 0
        assert head.symbol_index.lookup("fibonacci")

    def test_retrieval_by_symbol(self, sample_chunks: list[Chunk]) -> None:
        head = HeadA(sample_chunks)
        results = head.retrieve("fibonacci function", n_results=5)
        assert len(results) > 0
        chunk_ids = [r.chunk.chunk_id for r in results]
        assert "ch-001" in chunk_ids  # defines fibonacci

    def test_retrieval_scores_ordered(self, sample_chunks: list[Chunk]) -> None:
        head = HeadA(sample_chunks)
        results = head.retrieve("fibonacci function", n_results=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_n_results_limit(self, sample_chunks: list[Chunk]) -> None:
        head = HeadA(sample_chunks)
        results = head.retrieve("fibonacci", n_results=2)
        assert len(results) <= 2

    def test_retrieval_no_match(self, sample_chunks: list[Chunk]) -> None:
        head = HeadA(sample_chunks)
        results = head.retrieve("completely unrelated topic about cooking")
        assert len(results) == 0

    def test_head_name(self, sample_chunks: list[Chunk]) -> None:
        head = HeadA(sample_chunks)
        assert head.name == "head_a"

    def test_graph_expansion(self, sample_chunks: list[Chunk]) -> None:
        """main.py imports fibonacci — should get expanded as neighbor."""
        head = HeadA(sample_chunks)
        results = head.retrieve("parse_yaml config", n_results=5)
        chunk_ids = [r.chunk.chunk_id for r in results]
        assert "ch-002" in chunk_ids  # defines parse_yaml

    def test_deterministic(self, sample_chunks: list[Chunk]) -> None:
        head1 = HeadA(sample_chunks)
        head2 = HeadA(sample_chunks)
        r1 = head1.retrieve("fibonacci", n_results=5)
        r2 = head2.retrieve("fibonacci", n_results=5)
        assert [r.chunk.chunk_id for r in r1] == [r.chunk.chunk_id for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]
