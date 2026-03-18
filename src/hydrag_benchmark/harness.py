"""Combination harness — runs all head configurations and produces comparison matrix.

Per RFC §2.3, five configurations are tested on the same corpus/question set:
  A-only, B-only, C-only, A+B (union), A+B+C (full coverage)

Outputs a versioned JSON comparison matrix with per-config metrics.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .heads.base import Chunk, ScoredChunk
from .heads.head_a import HeadA
from .heads.head_b import HeadB
from .heads.head_c import HeadC
from .metrics import chunk_overlap, latency_stats, mrr, recall_at_1, recall_at_k
from .suite import BenchSuite

logger = logging.getLogger("hydrag_benchmark.harness")

CONFIGS: list[str] = ["A-only", "B-only", "C-only", "A+B", "A+B+C"]


@dataclass
class ConfigResult:
    """Results for a single configuration across all cases."""

    config_name: str
    cases: list[dict[str, object]] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)


@dataclass
class ComparisonMatrix:
    """Full comparison matrix across all configurations."""

    schema_version: str = "0.2"
    run_id: str = ""
    timestamp: str = ""
    tool_version: str = ""
    suite_name: str = ""
    corpus_dir: str = ""
    configs: list[ConfigResult] = field(default_factory=list)
    head_b_filter_stats: dict[str, int] = field(default_factory=dict)
    head_b_rejection_rate: float = 0.0


def _merge_results(
    *result_lists: list[ScoredChunk],
    n_results: int = 10,
) -> list[ScoredChunk]:
    """Merge results from multiple heads via score-based deduplication.

    Takes the union of results. If a chunk appears from multiple heads,
    keep the entry with the highest score.
    """
    best: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for sc in results:
            existing = best.get(sc.chunk.chunk_id)
            if existing is None or sc.score > existing.score:
                best[sc.chunk.chunk_id] = sc

    merged = sorted(best.values(), key=lambda x: x.score, reverse=True)
    return merged[:n_results]


def _run_config(
    config_name: str,
    query: str,
    head_a: HeadA,
    head_b: HeadB,
    head_c: HeadC,
    n_results: int,
) -> list[ScoredChunk]:
    """Execute a single configuration and return results."""
    if config_name == "A-only":
        return head_a.retrieve(query, n_results)
    elif config_name == "B-only":
        return head_b.retrieve(query, n_results)
    elif config_name == "C-only":
        return head_c.retrieve(query, n_results)
    elif config_name == "A+B":
        a_results = head_a.retrieve(query, n_results)
        b_results = head_b.retrieve(query, n_results)
        return _merge_results(a_results, b_results, n_results=n_results)
    elif config_name == "A+B+C":
        a_results = head_a.retrieve(query, n_results)
        b_results = head_b.retrieve(query, n_results)
        c_results = head_c.retrieve(query, n_results)
        return _merge_results(a_results, b_results, c_results, n_results=n_results)
    else:
        raise ValueError(f"Unknown config: {config_name}")


def run_multihead_benchmark(
    suite: BenchSuite,
    chunks: list[Chunk],
    head_a: HeadA,
    head_b: HeadB,
    head_c: HeadC,
    corpus_dir: str,
    n_results: int = 5,
    sidecar_path: Path | None = None,
) -> str:
    """Execute the full multi-head benchmark and return comparison matrix JSON.

    Runs all 5 configurations (A-only, B-only, C-only, A+B, A+B+C) against
    every case in the suite, computes metrics, and emits the comparison matrix.
    """
    matrix = ComparisonMatrix(
        run_id=f"mhbench-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool_version=__version__,
        suite_name=suite.name,
        corpus_dir=corpus_dir,
    )

    # Capture Head B filter stats
    if head_b.index is not None:
        matrix.head_b_filter_stats = dict(head_b.index.filter_stats)
        matrix.head_b_rejection_rate = round(head_b.filter_rejection_rate, 3)

    # Save question sidecar (DoD item 5)
    if sidecar_path:
        head_b.save_sidecar(sidecar_path)

    for config_name in CONFIGS:
        cfg_result = ConfigResult(config_name=config_name)

        case_latencies: list[float] = []
        for case in suite.cases:
            t0 = time.monotonic()
            error: str | None = None
            retrieved_texts: list[str] = []

            try:
                results = _run_config(
                    config_name, case.query, head_a, head_b, head_c, n_results,
                )
                retrieved_texts = [sc.chunk.text for sc in results]
            except Exception as exc:
                error = str(exc)

            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            case_latencies.append(elapsed_ms)

            cfg_result.cases.append({
                "case_id": case.id,
                "query": case.query,
                "config": config_name,
                "recall_at_1": recall_at_1(retrieved_texts, case.relevant_phrases),
                "recall_at_k": recall_at_k(retrieved_texts, case.relevant_phrases),
                "mrr": mrr(retrieved_texts, case.relevant_phrases),
                "chunk_overlap": chunk_overlap(retrieved_texts, case.relevant_phrases),
                "latency_ms": elapsed_ms,
                "error": error,
            })

        n = len(cfg_result.cases)
        cfg_result.summary = {
            "total_cases": n,
            "recall_at_1": round(sum(c["recall_at_1"] for c in cfg_result.cases) / n, 3) if n else 0.0,
            "recall_at_k": round(sum(c["recall_at_k"] for c in cfg_result.cases) / n, 3) if n else 0.0,
            "mrr": round(sum(c["mrr"] for c in cfg_result.cases) / n, 3) if n else 0.0,
            "chunk_overlap": round(sum(c["chunk_overlap"] for c in cfg_result.cases) / n, 3) if n else 0.0,
            "latency_ms": latency_stats(case_latencies),
        }
        matrix.configs.append(cfg_result)

    return json.dumps(_matrix_to_dict(matrix), indent=2)


def _matrix_to_dict(matrix: ComparisonMatrix) -> dict[str, object]:
    """Convert ComparisonMatrix to a serializable dict."""
    return {
        "schema_version": matrix.schema_version,
        "run_id": matrix.run_id,
        "timestamp": matrix.timestamp,
        "tool_version": matrix.tool_version,
        "suite_name": matrix.suite_name,
        "corpus_dir": matrix.corpus_dir,
        "head_b_filter_stats": matrix.head_b_filter_stats,
        "head_b_rejection_rate": matrix.head_b_rejection_rate,
        "configs": [
            {
                "config_name": cfg.config_name,
                "cases": cfg.cases,
                "summary": cfg.summary,
            }
            for cfg in matrix.configs
        ],
    }
