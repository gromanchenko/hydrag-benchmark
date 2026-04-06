"""Tests for canonical head IDs and legacy alias normalization."""

from __future__ import annotations

from hydrag_benchmark.heads.head_ids import CANONICAL_HEAD_IDS, canonicalize_head_ids, normalize_head_id


def test_normalize_head_id_supports_legacy_aliases() -> None:
    assert normalize_head_id("head_d") == "fts5_baseline"
    assert normalize_head_id("head_e") == "fts5_enriched"
    assert normalize_head_id("head_hydrag") == "hydrag_full"


def test_normalize_head_id_supports_canonical() -> None:
    assert normalize_head_id("fts5_baseline") == "fts5_baseline"
    assert normalize_head_id("fts5_enriched") == "fts5_enriched"
    assert normalize_head_id("symbol_graph") == "symbol_graph"


def test_canonicalize_head_ids_deduplicates_preserving_order() -> None:
    heads = ["head_d", "fts5_baseline", "head_e", "head_d", "hydrag_full"]
    assert canonicalize_head_ids(heads) == ["fts5_baseline", "fts5_enriched", "hydrag_full"]


def test_all_canonical_ids_are_normalizable() -> None:
    for head_id in CANONICAL_HEAD_IDS:
        assert normalize_head_id(head_id) == head_id
