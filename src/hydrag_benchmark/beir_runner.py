"""BEIR benchmark runner for all retrieval heads.

Loads a standard BEIR dataset, indexes corpus into the specified heads,
runs all queries, and computes doc-ID-based IR metrics: nDCG@10,
Recall@10, MRR@10, MAP@10.

Canonical heads:
    - symbol_graph: Graph/index retrieval (CPU-only)
    - dense_doc2query: Doc2Query + dense embedding (GPU optional)
    - hybrid_graph_dense: Hybrid rerank (symbol graph + dense)
    - fts5_baseline: SQLite FTS5 BM25 (CPU-only)
    - fts5_enriched: FTS5 + Ollama keyword enrichment
    - hydrag_full: Full HydRAG pipeline

Legacy aliases are still accepted: head_a/head_b/head_c/head_d/head_e/head_hydrag.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# T-975: default max corpus size for fts5_enriched Ollama enrichment.
# Ollama keyword enrichment throughput ~3 docs/min — above this threshold
# building the enriched index would take years on large corpora.  Configurable
# via HYDRAG_MAX_CORPUS_FOR_ENRICHMENT env var (0 = unlimited).
_DEFAULT_MAX_ENRICH_CORPUS: int = 500_000


# ── T-177: resource sampling helpers ─────────────────────────────────────────

def _peak_rss_mb() -> float:
    """Return peak RSS of this process in MB (POSIX; 0.0 on unsupported platforms)."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes
        if sys.platform == "darwin":
            return round(rss / (1024 * 1024), 1)
        return round(rss / 1024, 1)
    except Exception:  # pragma: no cover
        return 0.0


def _peak_vram_mb() -> float:
    """Return CUDA peak allocated VRAM in MB since last reset, or 0.0 if unavailable."""
    try:
        import torch  # noqa: PLC0415  (local import; torch may be absent)
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:  # pragma: no cover
        pass
    return 0.0


# ── T-977: DB snapshot manifest ──────────────────────────────────────────────

@dataclass
class DbSnapshotManifest:
    """Metadata sidecar written alongside a pre-seeded DB artifact on S3.

    schema_version 1 — all int fields are int64-compatible Python int.
    """
    schema_version: int
    indexed_at: str           # ISO-8601 UTC
    dataset: str              # BEIR dataset name (hotpotqa, scifact, …)
    head_name: str            # e.g. "fts5_baseline"
    backend: str              # "sqlite" | "surrealdb"
    run_id: str               # beir-XXXXXXXX from HeadResult
    artifact_name: str        # filename (no path)
    s3_uri: str               # s3://bucket/key of the artifact
    artifact_size_bytes: int  # compressed file size in bytes
    artifact_sha256: str      # SHA-256 hex of the compressed artifact (S-002)
    db_row_count: int         # actual COUNT(*) from the live table
    n_corpus: int             # corpus docs fed into build_index()
    build_duration_s: float   # wall-clock index time (= HeadResult.index_time_s)
    instance_type: str        # EC2 instance type (or "unknown")
    arch: str                 # platform.machine() — "x86_64" | "aarch64"
    hydrag_benchmark_version: str  # e.g. "0.5.17"
    ollama_model: str | None  # populated for enriched heads; None for baseline


def _get_hydrag_benchmark_version() -> str:
    """Return installed hydrag-benchmark package version, or 'dev' if uninstalled."""
    try:
        return importlib.metadata.version("hydrag-benchmark")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _get_ec2_instance_type() -> str:
    """Query EC2 IMDS for instance type; fall back to env var then 'unknown'."""
    env_val = os.environ.get("EC2_INSTANCE_TYPE", "")
    if env_val:
        return env_val
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "10"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode().strip()
    except Exception:
        return "unknown"


def _sqlite_row_count(db_path: Path) -> int:
    """Return COUNT(*) from chunks_fts in the SQLite file at db_path."""
    with sqlite3.connect(str(db_path)) as con:
        row = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
    return int(row[0]) if row else 0


def _gzip_sha256(src: Path, dst: Path) -> None:
    """Compress src → dst (gzip) writing atomically."""
    tmp = dst.with_suffix(".tmp")
    try:
        with open(src, "rb") as fin, gzip.open(tmp, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
        tmp.replace(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _upload_sqlite_snapshot(
    db_path: Path,
    *,
    dataset: str,
    head_result: "HeadResult",
    snapshot_bucket: str,
    ollama_model: str | None,
) -> str:
    """Gzip db_path, upload to S3, write manifest sidecar. Returns s3_uri.

    S3 layout:
        s3://<bucket>/beir-snapshots/<dataset>/<head_name>/<run_id>/<artifact_name>
        s3://<bucket>/beir-snapshots/<dataset>/<head_name>/<run_id>/manifest.json
    """
    try:
        import boto3  # noqa: PLC0415  (optional dep; present on bench workers)
    except ImportError as exc:
        raise RuntimeError("boto3 not installed — cannot upload snapshot") from exc

    run_id = head_result.run_id
    head_name = head_result.head

    artifact_name = f"{dataset}-{head_name}-{run_id}.db.gz"
    gz_path = db_path.parent / artifact_name
    _gzip_sha256(db_path, gz_path)

    artifact_sha256 = _sha256_file(gz_path)
    artifact_size_bytes = gz_path.stat().st_size
    db_row_count = _sqlite_row_count(db_path)

    s3_prefix = f"beir-snapshots/{dataset}/{head_name}/{run_id}"
    artifact_key = f"{s3_prefix}/{artifact_name}"
    manifest_key = f"{s3_prefix}/manifest.json"
    s3_uri = f"s3://{snapshot_bucket}/{artifact_key}"

    manifest = DbSnapshotManifest(
        schema_version=1,
        indexed_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        head_name=head_name,
        backend="sqlite",
        run_id=run_id,
        artifact_name=artifact_name,
        s3_uri=s3_uri,
        artifact_size_bytes=artifact_size_bytes,
        artifact_sha256=artifact_sha256,
        db_row_count=db_row_count,
        n_corpus=head_result.n_corpus,
        build_duration_s=head_result.index_time_s,
        instance_type=_get_ec2_instance_type(),
        arch=platform.machine(),
        hydrag_benchmark_version=_get_hydrag_benchmark_version(),
        ollama_model=ollama_model,
    )

    s3 = boto3.client("s3")
    logger.info("Uploading snapshot artifact: %s", s3_uri)
    s3.upload_file(str(gz_path), snapshot_bucket, artifact_key)
    s3.put_object(
        Bucket=snapshot_bucket,
        Key=manifest_key,
        Body=json.dumps(asdict(manifest), indent=2).encode(),
        ContentType="application/json",
    )
    logger.info(
        "Snapshot uploaded: %s  sha256=%s  rows=%d  size=%dB",
        s3_uri, artifact_sha256, db_row_count, artifact_size_bytes,
    )
    gz_path.unlink(missing_ok=True)
    return s3_uri


def _find_snapshot_db(
    snapshot_dir: Path,
    dataset: str,
    head_name: str,
) -> Path | None:
    """Find a pre-seeded .db file in snapshot_dir and verify via manifest.

    Expected layout (populated by AMI bake or manual download):
        <snapshot_dir>/<dataset>/<head_name>/<run_id>/manifest.json
        <snapshot_dir>/<dataset>/<head_name>/<run_id>/<artifact>.db.gz

    Returns the uncompressed .db path ready for HeadD, or None.
    """
    head_dir = snapshot_dir / dataset / head_name
    if not head_dir.is_dir():
        return None

    # Pick the newest run_id sub-directory (lexicographic sort on dirs).
    run_dirs = sorted(
        (d for d in head_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    if not run_dirs:
        return None

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt manifest at %s — skipping", manifest_path)
            continue

        artifact_name = meta.get("artifact_name", "")
        expected_sha = meta.get("artifact_sha256", "")
        gz_path = run_dir / artifact_name
        if not gz_path.exists():
            continue

        # S-002: verify SHA-256 before trusting
        actual_sha = _sha256_file(gz_path)
        if actual_sha != expected_sha:
            logger.warning(
                "SHA-256 mismatch for %s (expected=%s actual=%s) — skipping",
                gz_path, expected_sha, actual_sha,
            )
            continue

        # Decompress to a temp file alongside the .gz
        db_path = gz_path.with_suffix("")  # strips .gz
        if not db_path.exists():
            logger.info("Decompressing snapshot: %s", gz_path)
            tmp = db_path.with_suffix(".tmp")
            try:
                with gzip.open(gz_path, "rb") as fin, open(tmp, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                tmp.replace(db_path)
            except Exception:
                tmp.unlink(missing_ok=True)
                logger.exception("Failed to decompress %s", gz_path)
                continue

        logger.info(
            "Using pre-seeded snapshot: %s (sha256=%s rows=%s)",
            db_path, expected_sha[:12], meta.get("db_row_count", "?"),
        )
        return db_path

    return None


from .beir_loader import download_beir_dataset, load_beir_corpus, load_beir_qrels, load_beir_queries
from .heads.base import Chunk, RetrievalHead, ScoredChunk
from .sys_sampler import PhaseSampler
from .heads.head_a import HeadA
from .heads.head_b import HeadB
from .heads.head_c import HeadC
from .heads.head_d import HeadD
from .heads.head_d_chroma import HeadDChroma
from .heads.head_e import HeadE
from .heads.head_hydrag import HeadHydrag
from .heads.head_ids import canonicalize_head_ids, normalize_head_id

logger = logging.getLogger("hydrag_benchmark.beir_runner")


def _normalize_heads_with_warnings(heads: list[str]) -> list[str]:
    """Normalize head IDs and emit explicit warnings for dropped items."""
    normalized_heads = canonicalize_head_ids(heads)
    seen: set[str] = set()
    unknown_heads: list[str] = []
    duplicate_heads: list[str] = []

    for raw_head in heads:
        canonical = normalize_head_id(raw_head)
        if canonical is None:
            unknown_heads.append(raw_head)
            continue
        if canonical in seen:
            duplicate_heads.append(raw_head)
            continue
        seen.add(canonical)

    if unknown_heads:
        unknown_unique = list(dict.fromkeys(unknown_heads))
        logger.warning("Ignoring unknown heads: %s", ", ".join(unknown_unique))
    if duplicate_heads:
        duplicate_unique = list(dict.fromkeys(duplicate_heads))
        logger.warning(
            "Ignoring duplicate heads after normalization: %s",
            ", ".join(duplicate_unique),
        )

    return normalized_heads

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
class GpuInfo:
    enabled: bool
    name: str = ""
    index: int = 0
    vram_total_mb: int = 0
    vram_free_mb: int = 0


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
    # T-5052 A11/B-05: counts of which internal hydrag-core head actually
    # produced each retrieved result (from ScoredChunk.metadata, populated
    # only by HeadHydrag today), plus how many were fast-path/CRAG-skipped.
    # Empty/zero for heads that don't report this (Head D, Head E, ...) --
    # never fabricated. This is what makes a BM25-only "hydrag_full" run
    # visible instead of indistinguishable from a full-pipeline run.
    head_origin_counts: dict[str, int] = field(default_factory=dict)
    fast_path_count: int = 0
    crag_skipped_count: int = 0


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
    # T-5052 A11/B-05: aggregate of every query's head_origin_counts/
    # fast_path_count/crag_skipped_count, visible without inspecting each
    # individual query.
    head_origin_counts: dict[str, int] = field(default_factory=dict)
    fast_path_count: int = 0
    crag_skipped_count: int = 0
    # T-975: set to True when fts5_enriched was skipped due to corpus-size gate.
    # Downstream consumers should treat all metric fields as None/null when True.
    enrichment_skipped: bool = False
    # T-177: resource / throughput fields (0.0 = not measured).
    index_time_s: float = 0.0
    qps: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vram_mb: float = 0.0
    # T-1002: per-phase sys_metrics from PhaseSampler (CPU, RAM, disk, net, GPU).
    sys_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeirBenchmarkResult:
    dataset: str
    gpu: GpuInfo = field(default_factory=lambda: GpuInfo(enabled=False))
    heads: list[HeadResult] = field(default_factory=list)


# ── GPU detection ─────────────────────────────────────────────────────────────


def _detect_gpu() -> GpuInfo:
    """Detect GPU via torch.cuda (preferred) or nvidia-smi fallback."""
    # Try torch.cuda first
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            free, total = torch.cuda.mem_get_info(idx)
            return GpuInfo(
                enabled=True,
                name=props.name,
                index=idx,
                vram_total_mb=total // (1024 * 1024),
                vram_free_mb=free // (1024 * 1024),
            )
    except ImportError:
        pass

    # Fallback: nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                line = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                return GpuInfo(
                    enabled=True,
                    name=parts[1],
                    index=int(parts[0]),
                    vram_total_mb=int(parts[2]),
                    vram_free_mb=int(parts[3]),
                )
        except (subprocess.TimeoutExpired, IndexError, ValueError):
            pass

    return GpuInfo(enabled=False)


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
    head: RetrievalHead,
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

        # B-05: tally which internal head produced each result, plus
        # fast-path/CRAG-skip flags, from ScoredChunk.metadata (empty for
        # heads that don't report it -- never fabricated).
        query_head_origin_counts: dict[str, int] = {}
        query_fast_path_count = 0
        query_crag_skipped_count = 0
        for sc in scored_chunks:
            origin = sc.metadata.get("hydrag_head_origin")
            if origin:
                query_head_origin_counts[origin] = query_head_origin_counts.get(origin, 0) + 1
            if sc.metadata.get("fast_path_triggered"):
                query_fast_path_count += 1
            if sc.metadata.get("crag_skipped"):
                query_crag_skipped_count += 1

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
            head_origin_counts=query_head_origin_counts,
            fast_path_count=query_fast_path_count,
            crag_skipped_count=query_crag_skipped_count,
        ))

    n = len(results)
    total_latency_ms = sum(r.latency_ms for r in results)
    # T-177: QPS = n_queries / total_search_time_s (wall-clock serial queries).
    qps = round(n / (total_latency_ms / 1000.0), 2) if total_latency_ms > 0 else 0.0

    # B-05: aggregate per-query provenance across the whole head result.
    agg_head_origin_counts: dict[str, int] = {}
    for r in results:
        for origin, count in r.head_origin_counts.items():
            agg_head_origin_counts[origin] = agg_head_origin_counts.get(origin, 0) + count

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
        avg_latency_ms=round(total_latency_ms / n, 2) if n else 0.0,
        queries=results,
        qps=qps,
        head_origin_counts=agg_head_origin_counts,
        fast_path_count=sum(r.fast_path_count for r in results),
        crag_skipped_count=sum(r.crag_skipped_count for r in results),
    )


def run_beir_benchmark(
    dataset: str = "scifact",
    heads: list[str] | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    max_queries: int = 0,
    seed: int = 42,
    *,
    ollama_model: str = "qwen3:4b",
    ollama_host: str = "http://localhost:11434",
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
    use_gpu: bool = False,
    doc2query_model: str = "qwen3:4b",
    doc2query_api_url: str = "http://localhost:11434",
    # T-964: multi-backend comparison params
    surrealdb_url: str = "ws://localhost:8000/rpc",
    surrealdb_namespace: str = "hydrag_beir",
    surrealdb_username: str | None = None,
    surrealdb_password: str | None = None,
    ollama_embedding_model: str = "nomic-embed-text",
    # T-975: corpus-size gate for fts5_enriched (0 = use env var or module default)
    max_enrich_corpus: int = 0,
    # T-977: save pre-seeded DB artifacts to S3 after indexing
    save_snapshots: bool = False,
    snapshot_bucket: str = "lp2-artifacts",
    # T-977: load pre-seeded DB from local snapshot directory
    load_snapshot_dir: Path | None = None,
) -> BeirBenchmarkResult:
    """Run BEIR benchmark for canonical heads (legacy aliases accepted).

    Args:
        dataset: BEIR dataset name (e.g., "scifact", "nfcorpus").
        heads: Which heads to run. Default: ["fts5_baseline", "fts5_enriched"].
        cache_dir: Directory to cache downloaded BEIR datasets.
        output_dir: Directory to write JSON results.
        max_queries: Limit queries (0 = all), selected via a seeded random
            sample of the queries that have qrels, not the first N.
        seed: Random seed for max_queries sampling (B-02/B-03).
        ollama_model: Model for Head E enrichment.
        ollama_host: Ollama API endpoint.
        embedding_model: Dense embedding model for Head B/C.
        use_gpu: Use GPU-accelerated TransformersEmbedder for Head B/C.
        doc2query_model: Ollama model for Doc2Query generation (Head B).
        surrealdb_url: SurrealDB WebSocket URL for surreal_fts head.
        surrealdb_namespace: SurrealDB namespace for surreal_fts head.
        ollama_embedding_model: Ollama embedding model for chroma_vector head.
        doc2query_api_url: Ollama API URL for Doc2Query (Head B).

    Returns:
        BeirBenchmarkResult with per-head metrics.
    """
    if heads is None:
        heads = ["fts5_baseline", "fts5_enriched"]

    normalized_heads = _normalize_heads_with_warnings(heads)
    if not normalized_heads:
        logger.warning("No valid heads after normalization: %s", heads)
        return BeirBenchmarkResult(dataset=dataset)

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
        # B-02/B-03: sample with a seeded RNG over a deterministically
        # ordered candidate list, not a bare [:max_queries] slice -- the
        # latter always picked the same first N regardless of any seed,
        # and only ever exercised whichever queries happened to load
        # first. The corpus itself is never reduced here.
        candidate_qids = sorted(qid for qid in queries if qid in qrels)
        valid_qids = random.Random(seed).sample(
            candidate_qids, min(max_queries, len(candidate_qids))
        )
        queries = {qid: queries[qid] for qid in valid_qids}

    # Convert corpus to chunks
    chunks, chunk_to_doc = _corpus_to_chunks(corpus)
    n_corpus = len(chunks)
    logger.info("Corpus: %d docs, Queries: %d (with qrels)", n_corpus, len(queries))

    # T-975: resolve effective enrichment corpus-size gate
    _env_gate = int(os.environ.get("HYDRAG_MAX_CORPUS_FOR_ENRICHMENT", "0"))
    _effective_max_enrich: int = max_enrich_corpus or _env_gate or _DEFAULT_MAX_ENRICH_CORPUS

    # Lazy-init shared objects for head_b/c
    _embedder = None
    _doc2query = None

    def _get_embedder():
        nonlocal _embedder
        if _embedder is None:
            if use_gpu:
                from .embedding import EmbeddingConfig, TransformersEmbedder
                cfg = EmbeddingConfig(model_name=embedding_model)
                _embedder = TransformersEmbedder(cfg)
            else:
                from .embedding import HashEmbedder
                _embedder = HashEmbedder()
        return _embedder

    def _get_doc2query():
        nonlocal _doc2query
        if _doc2query is None:
            from .doc2query import Doc2QueryConfig, Doc2QueryGenerator
            cfg = Doc2QueryConfig(
                model=doc2query_model,
                api_url=doc2query_api_url,
            )
            _doc2query = Doc2QueryGenerator(cfg)
        return _doc2query

    # Detect GPU
    gpu_info = _detect_gpu()
    if gpu_info.enabled:
        logger.info("GPU detected: %s (index=%d, VRAM=%dMB total, %dMB free)",
                     gpu_info.name, gpu_info.index, gpu_info.vram_total_mb, gpu_info.vram_free_mb)
    else:
        logger.info("No GPU detected")

    result = BeirBenchmarkResult(dataset=dataset, gpu=gpu_info)

    for head_name in normalized_heads:
        logger.info("Running head: %s", head_name)
        closeable = True

        if head_name == "symbol_graph":
            # HeadA takes chunks in constructor (builds index internally)
            t0 = time.monotonic()
            head = HeadA(chunks)
            index_time = time.monotonic() - t0
            closeable = False
        elif head_name == "dense_doc2query":
            head = HeadB(
                embedder=_get_embedder(),
                doc2query=_get_doc2query(),
            )
        elif head_name == "hybrid_graph_dense":
            # HeadC needs HeadA + HeadB pre-built on the same corpus
            head_a_for_c = HeadA(chunks)
            head_b_for_c = HeadB(
                embedder=_get_embedder(),
                doc2query=_get_doc2query(),
            )
            head_b_for_c.build_index(chunks)
            head = HeadC(head_a=head_a_for_c, head_b=head_b_for_c)
            closeable = False
        elif head_name == "fts5_baseline":
            # T-977: check for a pre-seeded snapshot first.
            _snapshot_db_path: Path | None = None
            _snap_dir: Path | None = None
            if load_snapshot_dir is not None:
                _snapshot_db_path = _find_snapshot_db(load_snapshot_dir, dataset, "fts5_baseline")

            if _snapshot_db_path is not None:
                # Use the pre-seeded DB — skip build_index entirely.
                head = HeadD(db_path=_snapshot_db_path)
                logger.info("Loaded pre-seeded snapshot for fts5_baseline/%s — skipping build_index", dataset)
            elif save_snapshots:
                # T-977: use a real file path when snapshots are enabled so the
                # populated .db can be harvested after the benchmark run.
                _snap_dir = Path(tempfile.mkdtemp(prefix="beir-snap-"))
                _fts5_db_path = _snap_dir / f"{dataset}-fts5_baseline.db"
                head = HeadD(db_path=_fts5_db_path)
            else:
                _fts5_db_path = Path(":memory:")
                head = HeadD(db_path=_fts5_db_path)
        elif head_name == "fts5_enriched":
            # T-975: corpus-size gate — skip enrichment on large corpora.
            if _effective_max_enrich > 0 and n_corpus > _effective_max_enrich:
                logger.warning(
                    "fts5_enriched skipped: corpus %d docs > max_enrich_corpus %d "
                    "(HYDRAG_MAX_CORPUS_FOR_ENRICHMENT=%d). "
                    "Ollama enrichment at ~3 docs/min would take ~%d days. "
                    "Increase limit with --max-corpus-for-enrichment or set "
                    "HYDRAG_MAX_CORPUS_FOR_ENRICHMENT=0 to disable gate.",
                    n_corpus,
                    _effective_max_enrich,
                    _effective_max_enrich,
                    n_corpus // (3 * 60 * 24),
                )
                skipped_result = HeadResult(
                    run_id=f"beir-{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    head="fts5_enriched",
                    dataset=dataset,
                    n_queries=0,
                    n_corpus=n_corpus,
                    avg_ndcg_at_10=0.0,
                    avg_recall_at_10=0.0,
                    avg_mrr_at_10=0.0,
                    avg_map_at_10=0.0,
                    avg_latency_ms=0.0,
                    enrichment_skipped=True,
                )
                result.heads.append(skipped_result)
                continue
            head = HeadE(
                ollama_host=ollama_host,
                model=ollama_model,
            )
        elif head_name == "hydrag_full":
            head = HeadHydrag(
                ollama_host=ollama_host,
                ollama_model=ollama_model,
            )
        elif head_name == "surreal_fts":
            # T-964: SurrealDB disjunctive FTS head — database name is per-dataset
            # to keep index runs isolated when multiple datasets share the same instance.
            # T-5060: imported lazily -- surrealdb is not a declared dependency
            # of hydrag-benchmark (only hydrag-core[chromadb] is), so a
            # module-level import here would break every import of this
            # module whenever surrealdb isn't installed.
            from .heads.head_d_surreal import HeadDSurreal

            surreal_db_name = f"beir_{dataset}_{uuid.uuid4().hex[:8]}"
            head = HeadDSurreal(
                surrealdb_url=surrealdb_url,
                namespace=surrealdb_namespace,
                database=surreal_db_name,
                username=surrealdb_username,
                password=surrealdb_password,
            )
        elif head_name == "chroma_vector":
            # T-964: ChromaDB vector head — embeddings via Ollama.
            head = HeadDChroma(
                ollama_host=ollama_host,
                embedding_model=ollama_embedding_model,
            )
        else:
            logger.warning("Unknown head: %s, skipping", head_name)
            continue

        # Index (HeadA already indexed in constructor)
        _skip_index = False
        if head_name == "fts5_baseline" and _snapshot_db_path is not None:
            _skip_index = True

        t0 = time.monotonic()
        index_sampler_result = None
        query_sampler_result = None
        try:
            if _skip_index:
                # T-977: pre-seeded snapshot — populate dicts only, FTS5 index
                # is already in the .db file.  Saves ~100% of index_time.
                head.load_corpus_metadata(chunks)
                index_time = time.monotonic() - t0
                logger.info("Loaded snapshot metadata for %s in %.1fs (skipped FTS5 indexing)", head_name, index_time)
            elif head_name not in ("symbol_graph", "hybrid_graph_dense"):
                with PhaseSampler("index") as idx_sampler:
                    head.build_index(chunks)
                index_sampler_result = idx_sampler.result()
                index_time = index_sampler_result.duration_s
                logger.info("Indexed %d chunks in %.1fs (%s)", n_corpus, index_time, head_name)
            else:
                index_time = time.monotonic() - t0

            # Evaluate
            with PhaseSampler("query") as qry_sampler:
                head_result = _evaluate_head(
                    head, queries, qrels, chunk_to_doc, dataset, n_corpus,
                )
            query_sampler_result = qry_sampler.result()
            # T-177: attach resource metrics to the result.
            head_result.index_time_s = round(index_time, 3)
            head_result.peak_rss_mb = _peak_rss_mb()
            head_result.peak_vram_mb = _peak_vram_mb()
            # T-1002: attach per-phase sys_metrics.
            hw: dict[str, Any] = {}
            if index_sampler_result is not None:
                hw["index"] = index_sampler_result.to_dict()
            if query_sampler_result is not None:
                hw["query"] = query_sampler_result.to_dict()
            head_result.sys_metrics = hw
            result.heads.append(head_result)

            # Summary
            logger.info(
                "%s — nDCG@10: %.4f  Recall@10: %.4f  MRR@10: %.4f  MAP@10: %.4f"
                "  Avg latency: %.2fms  QPS: %.1f  index: %.1fs  RSS: %.0fMB  VRAM: %.0fMB",
                head_name,
                head_result.avg_ndcg_at_10,
                head_result.avg_recall_at_10,
                head_result.avg_mrr_at_10,
                head_result.avg_map_at_10,
                head_result.avg_latency_ms,
                head_result.qps,
                head_result.index_time_s,
                head_result.peak_rss_mb,
                head_result.peak_vram_mb,
            )
        finally:
            if closeable and hasattr(head, "close"):
                head.close()

        # T-977: upload SQLite snapshot AFTER close() so WAL is flushed.
        if save_snapshots and head_name == "fts5_baseline" and _snap_dir is not None:
            try:
                _upload_sqlite_snapshot(
                    _fts5_db_path,
                    dataset=dataset,
                    head_result=head_result,
                    snapshot_bucket=snapshot_bucket,
                    ollama_model=None,
                )
            except Exception:
                logger.exception("Snapshot upload failed for %s/%s — benchmark result unaffected", dataset, head_name)
            finally:
                shutil.rmtree(_snap_dir, ignore_errors=True)
                _snap_dir = None

    # Clean up shared embedder GPU memory
    if _embedder is not None and hasattr(_embedder, "unload"):
        _embedder.unload()

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
    g = result.gpu
    if g.enabled:
        print(f"GPU: {g.name} (index={g.index}, VRAM={g.vram_total_mb}MB total, {g.vram_free_mb}MB free)")
    else:
        print("GPU: none")
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
