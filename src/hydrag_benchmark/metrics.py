"""Metric calculators — frozen 0.1 metric set.

recall_at_1, recall_at_k, mrr, chunk_overlap, latency_ms.{avg,p50,p95,p99}
"""

from __future__ import annotations

import math


def _normalize(text: str) -> str:
    return text.lower().strip()


def recall_at_1(retrieved: list[str], relevant_phrases: list[str]) -> float:
    """1.0 if top result contains at least one relevant phrase."""
    if not retrieved or not relevant_phrases:
        return 0.0
    top = _normalize(retrieved[0])
    return 1.0 if any(_normalize(p) in top for p in relevant_phrases) else 0.0


def recall_at_k(retrieved: list[str], relevant_phrases: list[str]) -> float:
    """Fraction of relevant phrases found in any of the top-k chunks."""
    if not relevant_phrases:
        return 1.0
    found = sum(
        1
        for phrase in relevant_phrases
        if any(_normalize(phrase) in _normalize(chunk) for chunk in retrieved)
    )
    return round(found / len(relevant_phrases), 3)


def mrr(retrieved: list[str], relevant_phrases: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant chunk (0 if none)."""
    for rank, chunk in enumerate(retrieved, start=1):
        norm = _normalize(chunk)
        if any(_normalize(p) in norm for p in relevant_phrases):
            return round(1.0 / rank, 3)
    return 0.0


def chunk_overlap(retrieved: list[str], relevant_phrases: list[str]) -> float:
    """Average per-phrase token overlap across all retrieved chunks."""
    if not relevant_phrases or not retrieved:
        return 0.0
    all_text = " ".join(_normalize(c) for c in retrieved)
    scores: list[float] = []
    for phrase in relevant_phrases:
        tokens = set(_normalize(phrase).split())
        if not tokens:
            continue
        found = sum(1 for t in tokens if t in all_text)
        scores.append(found / len(tokens))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def latency_stats(latencies: list[float]) -> dict[str, float]:
    """Compute avg, p50, p95, p99 from a list of latency_ms values."""
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def _avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3)

    def _pct(xs: list[float], p: float) -> float:
        s = sorted(xs)
        idx = p / 100 * (len(s) - 1)
        lo, hi = int(idx), min(int(math.ceil(idx)), len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 3)

    return {
        "avg": _avg(latencies),
        "p50": _pct(latencies, 50),
        "p95": _pct(latencies, 95),
        "p99": _pct(latencies, 99),
    }
