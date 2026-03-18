"""Tests for prefill_doc2query runner function and 'prefill' CLI subcommand."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hydrag_benchmark.runner import prefill_doc2query
from hydrag_benchmark.cli import main as cli_main

_D2Q_PATCH = "hydrag_benchmark.doc2query.Doc2QueryGenerator"


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _write_corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a small corpus directory and return its path."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, content in files.items():
        (corpus / name).write_text(content, encoding="utf-8")
    return corpus


_SMALL_CORPUS = {
    "a.py": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
    "b.py": "class HttpClient:\n    def get(self, url: str) -> bytes:\n        pass\n",
    "c.go": "func ParseYAML(data []byte) (map[string]any, error) { return nil, nil }\n",
}


class _FakeDoc2Query:
    """Fake Doc2Query that returns synthetic questions without calling ollama."""

    def generate(self, text: str) -> list[str]:
        return [f"What does this function do: {text[:20]}?", "How is this implemented?"]


# ── prefill_doc2query tests ───────────────────────────────────────────────────


class TestPrefillDoc2Query:
    def test_returns_json_stats(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path, _SMALL_CORPUS)
        cache_dir = tmp_path / "cache"

        with patch(
            _D2Q_PATCH,
            return_value=_FakeDoc2Query(),
        ):
            result = prefill_doc2query(corpus_dir=corpus, cache_dir=cache_dir)

        stats = json.loads(result)
        assert "total_chunks" in stats
        assert stats["total_chunks"] > 0
        assert stats["generated"] > 0
        assert stats["failed"] == 0
        assert stats["skipped_cached"] == 0

    def test_cache_dir_created(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path, _SMALL_CORPUS)
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()

        with patch(
            _D2Q_PATCH,
            return_value=_FakeDoc2Query(),
        ):
            prefill_doc2query(corpus_dir=corpus, cache_dir=cache_dir)

        assert cache_dir.exists()
        assert (cache_dir / "augmentation_cache.json").exists()

    def test_skips_cached_chunks(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path, _SMALL_CORPUS)
        cache_dir = tmp_path / "cache"

        call_count = 0

        class _CountingDoc2Query:
            def generate(self, text: str) -> list[str]:
                nonlocal call_count
                call_count += 1
                return ["question one", "question two"]

        # First pass: populate cache
        with patch(_D2Q_PATCH, return_value=_CountingDoc2Query()):
            result1 = prefill_doc2query(corpus_dir=corpus, cache_dir=cache_dir)

        first_calls = call_count
        assert first_calls > 0

        call_count = 0  # reset

        # Second pass: everything should be cached
        with patch(_D2Q_PATCH, return_value=_CountingDoc2Query()):
            result2 = prefill_doc2query(corpus_dir=corpus, cache_dir=cache_dir)

        assert call_count == 0, "Expected zero LLM calls on second pass (cache populated)"
        stats = json.loads(result2)
        assert stats["skipped_cached"] == json.loads(result1)["total_chunks"]

    def test_failed_chunks_marked_in_cache(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path, {"fail.py": "x = 1\n"})
        cache_dir = tmp_path / "cache"

        class _FailingDoc2Query:
            def generate(self, text: str) -> list[str]:
                raise RuntimeError("API down")

        with patch(_D2Q_PATCH,
                   return_value=_FailingDoc2Query()):
            result = prefill_doc2query(corpus_dir=corpus, cache_dir=cache_dir)

        stats = json.loads(result)
        assert stats["failed"] > 0
        assert stats["generated"] == 0

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        empty_corpus = tmp_path / "empty"
        empty_corpus.mkdir()

        with pytest.raises(RuntimeError, match="No indexable files"):
            prefill_doc2query(corpus_dir=empty_corpus)

    def test_no_cache_dir_still_works(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path, {"f.py": "def go(): pass\n"})

        with patch(
            _D2Q_PATCH,
            return_value=_FakeDoc2Query(),
        ):
            result = prefill_doc2query(corpus_dir=corpus, cache_dir=None)

        stats = json.loads(result)
        assert stats["cache_path"] is None
        assert stats["generated"] >= 0


# ── CLI prefill subcommand tests ──────────────────────────────────────────────


class TestCliPrefill:
    def test_prefill_subcommand_exists(self) -> None:
        # Should exit 0 on --help without error
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["prefill", "--help"])
        assert exc_info.value.code == 0

    def test_prefill_runs_and_prints_json(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        corpus = _write_corpus(tmp_path, {"m.py": "class MyModel: pass\n"})
        cache_dir = tmp_path / "cache"

        with patch(
            _D2Q_PATCH,
            return_value=_FakeDoc2Query(),
        ):
            rc = cli_main([
                "prefill",
                "--corpus-dir", str(corpus),
                "--cache-dir", str(cache_dir),
            ])

        assert rc == 0
        captured = capsys.readouterr()
        stats = json.loads(captured.out)
        assert "total_chunks" in stats

    def test_prefill_missing_corpus(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = cli_main(["prefill", "--corpus-dir", str(tmp_path / "nonexistent")])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
