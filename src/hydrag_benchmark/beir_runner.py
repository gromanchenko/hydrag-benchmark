"""BEIR benchmark runner for FTS5 Heads (D and E).

Loads a standard BEIR dataset, indexes corpus into Head D (raw FTS5)
and/or Head E (FTS5 + LLM enrichment), runs all queries, and computes
doc-ID-based IR metrics: nDCG@10, Recall@10, MRR@10, MAP@10.

Results are directly comparable with the docker/bench-ubuntu baseline:
  - similarity: nDCG@10 = 0.4796
  - hybrid:     nDCG@10 = 0.5788
  - hydrag:     nDCG@10 = 0.5875
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .beir_loader import download_beir_dataset, load_beir_corpus, load_beir_qrels, load_beir_queries
from .heads.base import Chunk, ScoredChunk
from .heads.head_d import HeadD
from .heads.head_e import HeadE

logger = logging.getLogger("hydrag_benchmark.beir_runner")

# ── Standard IR Metrics (doc-ID based, BEIR-compatible) ─────────────────────


def ndcg_at_k(retrieved_ids: list[str], qrel: dict[str, int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        rel = qrel.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    ideal_rels = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal_rels, 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], qrel: dict[str, int], k: int = 10) -> float:
    """Recall @ k."""
    if not qrel:
        return 0.0
    found = sum(1 for did in retrieved_ids[:k] if did in qrel)
    return found / len(qrel)


def mrr_at_k(retrieved_ids: list[str], qrel: dict[str, int], k: int = 10) -> float:
    """Mean Reciprocal Rank @ k."""
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in qrel:
            return 1.0 / i
    return 0.0


def map_at_k(retrieved_ids: list[str], qrel: dict[str, int], k: int = 10) -> float:
    """Mean Average Precision @ k."""
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in qrel:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(qrel), k) if qrel else 0.0


# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class QueryResult:
    query_id: str
    query: str
    head: str
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    map_at_10: float
    latency_ms: float
    n_relevant: int
    n_retrieved_relevant: int


@dataclass
class HeadResult:
    run_id: str
    timestamp: str
    head: str
    dataset: str
    n_queries: int
    n_corpus: int
    avg_ndcg_at_10: float
    avg_recall_at_10: float
    avg_mrr_at_10: float
    avg_map_at_10: float
    avg_latency_ms: float
    queries: list[QueryResult] = field(default_factory=list)


@dataclass
class BeirBenchmarkResult:
    dataset: str
    heads: list[HeadResult] = field(default_factory=list)


# ── Core runner ───────────────────────────────────────────────────────────────


def _corpus_to_chunks(corpus: dict[str, dict[str, str]]) -> tuple[list[Chunk], dict[str, str]]:
    """Convert BEIR corpus to benchmark Chunks. Returns (chunks, chunk_id→doc_id)."""
    chunks: list[Chunk] = []
    chunk_to_doc: dict[str, str] = {}
    for doc_id, doc in corpus.items():
        text = f"{doc['title']}\n{doc['text']}".strip()
        if not text:
            continue
        chunk_id = f"doc-{doc_id}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=text,
            source=f"beir:{doc_id}",
            symbols=[],
        ))
        chunk_to_doc[chunk_id] = doc_id
    return chunks, chunk_to_doc


def _evaluate_head(
    head: HeadD | HeadE,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    chunk_to_doc: dict[str, str],
    dataset_name: str,
    n_corpus: int,
    k: int = 10,
) -> HeadResult:
    """Run all queries through a head and compute IR metrics."""
    results: list[QueryResult] = []

    for qid, qtext in queries.items():
        if qid not in qrels:
            continue
        qrel = qrels[qid]

        t0 = time.monotonic()
        scored_chunks = head.retrieve(qtext, n_results=k)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

        # Map chunk_ids back to BEIR doc_ids (dedup, preserve ranking)
        seen: set[str] = set()
        doc_ids: list[str] = []
        for sc in scored_chunks:
            did = chunk_to_doc.get(sc.chunk.chunk_id, "")
            if did and did not in seen:
                seen.add(did)
                doc_ids.append(did)

        n_hit = sum(1 for did in doc_ids[:k] if did in qrel)

        results.append(QueryResult(
            query_id=qid,
            query=qtext,
            head=head.name,
            ndcg_at_10=ndcg_at_k(doc_ids, qrel, k),
            recall_at_10=recall_at_k(doc_ids, qrel, k),
            mrr_at_10=mrr_at_k(doc_ids, qrel, k),
            map_at_10=map_at_k(doc_ids, qrel, k),
            latency_ms=elapsed_ms,
            n_relevant=len(qrel),
            n_retrieved_relevant=n_hit,
        ))

    n = len(results)
    return HeadResult(
        run_id=f"beir-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        head=head.name,
        dataset=dataset_name,
        n_queries=n,
        n_corpus=n_corpus,
        avg_ndcg_at_10=round(sum(r.ndcg_at_10 for r in results) / n, 4) if n else 0.0,
        avg_recall_at_10=round(sum(r.recall_at_10 for r in results) / n, 4) if n else 0.0,
        avg_mrr_at_10=round(sum(r.mrr_at_10 for r in results) / n, 4) if n else 0.0,
        avg_map_at_10=round(sum(r.map_at_10 for r in results) / n, 4) if n else 0.0,
        avg_latency_ms=round(sum(r.latency_ms for r in results) / n, 2) if n else 0.0,
        queries=results,
    )


def run_beir_benchmark(
    dataset: str = "scifact",
    heads: list[str] | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    max_queries: int = 0,
    *,
    ollama_model: str = "qwen3:4b",
    ollama_host: str = "http://localhost:11434",
) -> BeirBenchmarkResult:
    """Run BEIR benchmark for FTS5 heads.

    Args:
        dataset: BEIR dataset name (e.g., "scifact", "nfcorpus").
        heads: Which heads to run. Default: ["head_d", "head_e"].
        cache_dir: Directory to cache downloaded BEIR datasets.
        output_dir: Directory to write JSON results.
        max_queries: Limit queries (0 = all).
        ollama_model: Model for Head E enrichment.
        ollama_host: Ollama API endpoint.

    Returns:
        BeirBenchmarkResult with per-head metrics.
    """
    if heads is None:
        heads = ["head_d", "head_e"]

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "beir"
    if output_dir is None:
        output_dir = Path("beir-results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download / load dataset
    logger.info("Loading BEIR dataset: %s", dataset)
    dataset_dir = download_beir_dataset(dataset, cache_dir)
    corpus = load_beir_corpus(dataset_dir)
    queries = load_beir_queries(dataset_dir)
    qrels = load_beir_qrels(dataset_dir)

    if max_queries > 0:
        # Filter to queries that have qrels
        valid_qids = [qid for qid in queries if qid in qrels][:max_queries]
        queries = {qid: queries[qid] for qid in valid_qids}

    # Convert corpus to chunks
    chunks, chunk_to_doc = _corpus_to_chunks(corpus)
    n_corpus = len(chunks)
    logger.info("Corpus: %d docs, Queries: %d (with qrels)", n_corpus, len(queries))

    result = BeirBenchmarkResult(dataset=dataset)

    for head_name in heads:
        logger.info("Running head: %s", head_name)

        if head_name == "head_d":
            head = HeadD()
        elif head_name == "head_e":
            head = HeadE(
                ollama_host=ollama_host,
                model=ollama_model,
            )
        else:
            logger.warning("Unknown head: %s, skipping", head_name)
            continue

        # Index
        t0 = time.monotonic()
        head.build_index(chunks)
        index_time = time.monotonic() - t0
        logger.info("Indexed %d chunks in %.1fs (%s)", n_corpus, index_time, head_name)

        # Evaluate
        head_result = _evaluate_head(
            head, queries, qrels, chunk_to_doc, dataset, n_corpus,
        )
        result.heads.append(head_result)

        # Summary
        logger.info(
            "%s — nDCG@10: %.4f  Recall@10: %.4f  MRR@10: %.4f  MAP@10: %.4f  Avg latency: %.2fms",
            head_name,
            head_result.avg_ndcg_at_10,
            head_result.avg_recall_at_10,
            head_result.avg_mrr_at_10,
            head_result.avg_map_at_10,
            head_result.avg_latency_ms,
        )

        head.close()

    # Write results
    out_path = output_dir / f"beir-{dataset}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    logger.info("Results written to %s", out_path)

    # Print comparison table
    _print_comparison(result)

    return result


def _print_comparison(result: BeirBenchmarkResult) -> None:
    """Print a formatted comparison table to stdout."""
    print(f"\n{'=' * 70}")
    print(f"BEIR Benchmark: {result.dataset}")
    print(f"{'=' * 70}")
    print(f"{'Head':<10} {'nDCG@10':>10} {'Recall@10':>10} {'MRR@10':>10} {'MAP@10':>10} {'Latency':>10}")
    print(f"{'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for hr in result.heads:
        print(
            f"{hr.head:<10} {hr.avg_ndcg_at_10:>10.4f} {hr.avg_recall_at_10:>10.4f} "
            f"{hr.avg_mrr_at_10:>10.4f} {hr.avg_map_at_10:>10.4f} {hr.avg_latency_ms:>8.2f}ms"
        )
    # Reference baselines (from docker/bench-ubuntu SciFact run)
    if result.dataset == "scifact":
        print(f"\nReference baselines (ChromaDB, SciFact):")
        print(f"  similarity: nDCG@10 = 0.4796")
        print(f"  hybrid:     nDCG@10 = 0.5788")
        print(f"  hydrag:     nDCG@10 = 0.5875")
    print(f"{'=' * 70}\n")
