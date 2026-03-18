"""Tests for Doc2Query adaptive n_questions, smart_truncate, and config fingerprint."""

from __future__ import annotations

import pytest

from hydrag_benchmark.doc2query import (
    Doc2QueryConfig,
    Doc2QueryGenerator,
    compute_adaptive_n,
    smart_truncate,
)


# ── compute_adaptive_n ───────────────────────────────────────────────────────


class TestComputeAdaptiveN:
    """RFC §2.3 lookup table: <50→1-2, 50-200→3, 200-500→5, >500→7."""

    def test_tiny_chunk(self) -> None:
        # < 50 tokens → up to 2
        text = "short chunk with few words"
        assert compute_adaptive_n(text, max_questions=12) <= 2
        assert compute_adaptive_n(text, max_questions=12) >= 1

    def test_small_chunk(self) -> None:
        # 50–200 tokens → 3
        text = " ".join(f"word{i}" for i in range(100))
        assert compute_adaptive_n(text, max_questions=12) == 3

    def test_medium_chunk(self) -> None:
        # 200–500 tokens → 5
        text = " ".join(f"token{i}" for i in range(300))
        assert compute_adaptive_n(text, max_questions=12) == 5

    def test_large_chunk(self) -> None:
        # > 500 tokens → 7
        text = " ".join(f"token{i}" for i in range(600))
        assert compute_adaptive_n(text, max_questions=12) == 7

    def test_cap_at_max_questions(self) -> None:
        # Large chunk wants 7, but max_questions=4
        text = " ".join(f"token{i}" for i in range(600))
        assert compute_adaptive_n(text, max_questions=4) == 4

    def test_cap_at_max_questions_small(self) -> None:
        # Medium chunk wants 5, but max_questions=2
        text = " ".join(f"token{i}" for i in range(300))
        assert compute_adaptive_n(text, max_questions=2) == 2

    def test_floor_at_one(self) -> None:
        assert compute_adaptive_n("", max_questions=12) >= 1
        assert compute_adaptive_n("x", max_questions=0) >= 1

    def test_boundary_50_tokens(self) -> None:
        # Exactly 50 tokens should hit 50-200 bucket → 3
        text = " ".join(f"w{i}" for i in range(50))
        assert compute_adaptive_n(text, max_questions=12) == 3

    def test_boundary_200_tokens(self) -> None:
        # Exactly 200 tokens should hit 200-500 bucket → 5
        text = " ".join(f"w{i}" for i in range(200))
        assert compute_adaptive_n(text, max_questions=12) == 5

    def test_boundary_500_tokens(self) -> None:
        # Exactly 500 tokens should hit >500 bucket → 7
        text = " ".join(f"w{i}" for i in range(500))
        assert compute_adaptive_n(text, max_questions=12) == 7

    def test_49_tokens(self) -> None:
        # 49 tokens → <50 bucket → 2
        text = " ".join(f"w{i}" for i in range(49))
        assert compute_adaptive_n(text, max_questions=12) == 2


# ── smart_truncate ───────────────────────────────────────────────────────────


class TestSmartTruncate:
    def test_short_text_unchanged(self) -> None:
        text = "Short text under limit."
        assert smart_truncate(text, max_chars=4000) == text

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 4000
        assert smart_truncate(text, max_chars=4000) == text

    def test_paragraph_boundary_cut(self) -> None:
        para1 = "First paragraph. " * 100  # ~1700 chars
        para2 = "Second paragraph. " * 100
        para3 = "Third paragraph. " * 100
        text = f"{para1}\n\n{para2}\n\n{para3}"
        result = smart_truncate(text, max_chars=4000, overlap=200)
        # Should cut at a \n\n boundary
        assert "\n[...]\n" in result
        # Tail snippet should be present
        assert result.endswith(text[-200:])

    def test_sentence_boundary_fallback(self) -> None:
        # No \n\n before limit, but has ". " boundaries
        text = "Sentence one. " * 300  # ~4200 chars, no \n\n
        result = smart_truncate(text, max_chars=4000, overlap=200)
        # Should cut at a sentence boundary
        main_part = result.split("\n[...]\n")[0]
        assert main_part.rstrip().endswith(".")

    def test_hard_cut_fallback(self) -> None:
        # No \n\n and no ". " → hard cut at max_chars
        text = "x" * 5000
        result = smart_truncate(text, max_chars=4000, overlap=200)
        assert "\n[...]\n" in result
        main_part = result.split("\n[...]\n")[0]
        assert len(main_part) == 4000

    def test_tail_overlap_present(self) -> None:
        tail_content = "TAIL_MARKER " * 20
        text = "x" * 5000 + tail_content
        result = smart_truncate(text, max_chars=4000, overlap=200)
        # Last 200 chars of original should be in the result
        assert text[-200:] in result

    def test_no_tail_when_text_barely_exceeds(self) -> None:
        # len(text) <= max_chars + overlap → no tail
        text = "x" * 4100
        result = smart_truncate(text, max_chars=4000, overlap=200)
        assert "\n[...]\n" not in result or text[-200:] not in result.split("\n[...]\n")[-1]

    def test_custom_max_chars(self) -> None:
        text = "word " * 500  # 2500 chars
        result = smart_truncate(text, max_chars=1000, overlap=100)
        assert len(result) < len(text)

    def test_overlap_zero(self) -> None:
        text = "x" * 5000
        result = smart_truncate(text, max_chars=4000, overlap=0)
        # No tail → just the cut portion
        assert len(result) <= 4000


# ── Doc2QueryConfig ──────────────────────────────────────────────────────────


class TestDoc2QueryConfig:
    def test_fingerprint_includes_adaptive_n(self) -> None:
        c1 = Doc2QueryConfig(adaptive_n=False)
        c2 = Doc2QueryConfig(adaptive_n=True)
        assert c1.config_fingerprint() != c2.config_fingerprint()

    def test_fingerprint_includes_max_questions(self) -> None:
        c1 = Doc2QueryConfig(max_questions_per_chunk=12)
        c2 = Doc2QueryConfig(max_questions_per_chunk=8)
        assert c1.config_fingerprint() != c2.config_fingerprint()

    def test_fingerprint_stable(self) -> None:
        c = Doc2QueryConfig()
        assert c.config_fingerprint() == c.config_fingerprint()

    def test_default_adaptive_n_false(self) -> None:
        c = Doc2QueryConfig()
        assert c.adaptive_n is False

    def test_default_max_questions(self) -> None:
        c = Doc2QueryConfig()
        assert c.max_questions_per_chunk == 12


# ── Doc2QueryGenerator integration ──────────────────────────────────────────


class TestDoc2QueryGeneratorBuildPrompt:
    def test_smart_truncate_used(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig())
        long_text = "x" * 6000
        prompt = gen._build_prompt(long_text)
        # Should NOT contain the full 6000 chars
        assert len(prompt) < 6000 + 500  # prompt template overhead
        # Should contain the truncation marker
        assert "[...]" in prompt

    def test_adaptive_n_in_prompt(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig(adaptive_n=True))
        # Large chunk → should ask for 7 questions
        big_text = " ".join(f"word{i}" for i in range(600))
        prompt = gen._build_prompt(big_text, n=7)
        assert "exactly 7 questions" in prompt

    def test_non_adaptive_uses_baseline(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig(n_questions=4, adaptive_n=False))
        prompt = gen._build_prompt("some text")
        assert "exactly 4 questions" in prompt

    def test_custom_prompt_prepended(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig(custom_prompt="CONTEXT: K8s docs"))
        prompt = gen._build_prompt("chunk text")
        assert prompt.startswith("CONTEXT: K8s docs\n\n")
