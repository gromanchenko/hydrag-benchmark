"""Benchmark runner — single-strategy, single-suite local execution.

Indexes a corpus directory into ChromaDB, executes queries from a suite YAML,
computes frozen 0.1 metrics, and emits versioned JSON output.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .metrics import chunk_overlap, latency_stats, mrr, recall_at_1, recall_at_k
from .suite import BenchSuite

logger = logging.getLogger("hydrag_benchmark")

SCHEMA_VERSION = "0.1"


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    case_id: str
    query: str
    strategy: str
    recall_at_1: float
    recall_at_k: float
    mrr: float
    chunk_overlap: float
    latency_ms: float
    error: str | None = None


@dataclass
class RunResult:
    schema_version: str
    run_id: str
    timestamp: str
    tool_version: str
    suite_name: str
    strategy: str
    n_results: int
    seed: int
    corpus_dir: str
    cases: list[CaseResult]
    summary: dict[str, Any] = field(default_factory=dict)
    strategy_note: str = ""


# ── Corpus indexer ───────────────────────────────────────────────────────────

_CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".go", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".java", ".js", ".jsx", ".ts", ".tsx", ".rs", ".rb",
    ".scala", ".kt", ".swift", ".md", ".yaml", ".yml",
    ".toml", ".json", ".sh", ".bash", ".tf",
})


def _build_kb(
    corpus_dir: Path,
    db_path: Path,
    embedding_model: str,
) -> Any:
    """Build a ChromaDB-backed KnowledgeBase from a corpus directory.

    Requires hydrag-core[chromadb] — chromadb is an optional extra.
    """
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is required for benchmarking. "
            "Install with: pip install hydrag-core[chromadb]"
        ) from exc

    client = chromadb.PersistentClient(path=str(db_path))
    collection_name = "bench_corpus"

    # Reset if exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    files = [
        fp
        for fp in corpus_dir.rglob("*")
        if fp.is_file() and fp.suffix in _CODE_EXTENSIONS
    ]
    logger.info("Indexing %d files from %s", len(files), corpus_dir)

    doc_ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", fp, exc)
            continue
        if not text.strip():
            continue
        rel = str(fp.relative_to(corpus_dir))
        # Simple chunking: split by double newlines, cap at ~2000 chars
        chunks = _chunk_text(text, rel)
        for chunk_id, chunk_text in chunks:
            doc_ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({"source": rel})

    # Batch add (chromadb batch limit is ~41666)
    batch_size = 5000
    for i in range(0, len(doc_ids), batch_size):
        collection.add(
            ids=doc_ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    logger.info("Indexed %d chunks from %d files", len(doc_ids), len(files))
    return collection


def _chunk_text(text: str, source: str, max_chars: int = 2000) -> list[tuple[str, str]]:
    """Split text into chunks by double-newline paragraphs, capped at max_chars."""
    paragraphs = text.split("\n\n")
    chunks: list[tuple[str, str]] = []
    current = ""
    idx = 0
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append((f"{source}::chunk-{idx}", current.strip()))
            idx += 1
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append((f"{source}::chunk-{idx}", current.strip()))
    return chunks


# ── ChromaDB adapter ──────────────────────────────────────────────────────────

SUPPORTED_STRATEGIES: frozenset[str] = frozenset({
    "similarity", "hybrid", "crag", "hydrag",
})

_STRATEGY_HEADS: dict[str, set[str]] = {
    "hybrid": {"head_1"},
    "crag": {"head_1", "head_2_crag", "head_3a_semantic"},
    # "hydrag" uses all defaults (no heads= kwarg)
}


class _ChromaDBAdapter:
    """VectorStoreAdapter wrapping a chromadb.Collection.

    ChromaDB has no native BM25, so all three protocol methods
    delegate to the same collection.query(). This is intentional:
    benchmarks measure pipeline architecture (RRF fusion, head gating,
    CRAG judgment), not the underlying store.
    """

    def __init__(self, collection: Any, n_results: int = 5) -> None:
        self._collection = collection
        self._n_results = n_results

    def _query(self, query: str, n_results: int) -> list[str]:
        results = self._collection.query(query_texts=[query], n_results=n_results)
        return results.get("documents", [[]])[0]

    def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
        return self._query(query, n_results)

    def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
        return self._query(query, n_results)

    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
        return self._query(query, n_results)


# ── Strategy dispatch ────────────────────────────────────────────────────────


def _search_fn(strategy: str, collection: Any, n_results: int) -> Any:
    """Return a callable(query) -> list[str] for the given strategy.

    Strategies:
        similarity — Direct chromadb query (baseline, no hydrag-core).
        hybrid     — hydrag_search with head_1 only (no CRAG/fallbacks).
        crag       — hydrag_search with heads 1 + 2 + 3a.
        hydrag     — Full hydrag_search pipeline (all default heads).
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_STRATEGIES))}"
        )

    if strategy == "similarity":
        def _baseline(query: str) -> list[str]:
            results = collection.query(query_texts=[query], n_results=n_results)
            return results.get("documents", [[]])[0]
        return _baseline

    from hydrag import HydRAGConfig, hydrag_search

    adapter = _ChromaDBAdapter(collection, n_results)
    cfg = HydRAGConfig()
    heads = _STRATEGY_HEADS.get(strategy)

    def _dispatch(query: str) -> list[str]:
        results = hydrag_search(
            adapter,
            query,
            n_results=n_results,
            config=cfg,
            heads=heads,
        )
        return [r.text for r in results]

    return _dispatch


# ── Main runner ──────────────────────────────────────────────────────────────


def run_benchmark(
    suite_path: Path,
    strategy: str,
    corpus_dir: Path,
    n_results: int = 5,
    seed: int = 42,
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
    db_path: Path | None = None,
) -> str:
    """Execute a benchmark suite and return frozen 0.1 JSON string."""
    suite = BenchSuite.from_yaml(
        suite_path,
        strategy_override=strategy,
        n_results_override=n_results,
        seed_override=seed,
    )

    if db_path is None:
        db_path = Path(tempfile.mkdtemp(prefix="hydrag_bench_"))

    collection = _build_kb(corpus_dir, db_path, embedding_model)
    search = _search_fn(strategy, collection, n_results)

    case_results: list[CaseResult] = []
    for case in suite.cases:
        t0 = time.monotonic()
        error: str | None = None
        retrieved: list[str] = []

        try:
            retrieved = search(case.query)
        except Exception as exc:
            error = str(exc)

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        case_results.append(
            CaseResult(
                case_id=case.id,
                query=case.query,
                strategy=strategy,
                recall_at_1=recall_at_1(retrieved, case.relevant_phrases),
                recall_at_k=recall_at_k(retrieved, case.relevant_phrases),
                mrr=mrr(retrieved, case.relevant_phrases),
                chunk_overlap=chunk_overlap(retrieved, case.relevant_phrases),
                latency_ms=elapsed_ms,
                error=error,
            )
        )

    # Build summary
    latencies = [c.latency_ms for c in case_results]
    n = len(case_results)
    summary = {
        "total_cases": n,
        "recall_at_1": round(sum(c.recall_at_1 for c in case_results) / n, 3) if n else 0.0,
        "recall_at_k": round(sum(c.recall_at_k for c in case_results) / n, 3) if n else 0.0,
        "mrr": round(sum(c.mrr for c in case_results) / n, 3) if n else 0.0,
        "chunk_overlap": round(sum(c.chunk_overlap for c in case_results) / n, 3) if n else 0.0,
        "latency_ms": latency_stats(latencies),
    }

    _STRATEGY_NOTES = {
        "similarity": "",
        "hybrid": "ChromaDB-backed: semantic/keyword/hybrid all delegate to the same "
                  "collection.query(). Differences measure pipeline architecture "
                  "(RRF fusion, head gating), not underlying retrieval.",
        "crag": "ChromaDB-backed: semantic/keyword/hybrid all delegate to the same "
                "collection.query(). Differences measure CRAG gating overhead, "
                "not retrieval strategy.",
        "hydrag": "ChromaDB-backed: semantic/keyword/hybrid all delegate to the same "
                  "collection.query(). Differences measure full pipeline overhead "
                  "(head gating, CRAG, fallbacks), not underlying retrieval.",
    }

    result = RunResult(
        schema_version=SCHEMA_VERSION,
        run_id=f"bench-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool_version=__version__,
        suite_name=suite.name,
        strategy=strategy,
        n_results=n_results,
        seed=seed,
        corpus_dir=str(corpus_dir),
        cases=case_results,
        summary=summary,
        strategy_note=_STRATEGY_NOTES.get(strategy, ""),
    )

    return json.dumps(asdict(result), indent=2, ensure_ascii=False)


# ── Multi-head runner ────────────────────────────────────────────────────────


def _chunk_corpus(corpus_dir: Path) -> list["Chunk"]:
    """Parse a corpus directory into content-addressed Chunks for multi-head."""
    from .heads.base import Chunk

    files = [
        fp
        for fp in corpus_dir.rglob("*")
        if fp.is_file() and fp.suffix in _CODE_EXTENSIONS
    ]
    logger.info("Chunking %d files from %s", len(files), corpus_dir)

    chunks: list[Chunk] = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", fp, exc)
            continue
        if not text.strip():
            continue
        rel = str(fp.relative_to(corpus_dir))
        for _, chunk_text in _chunk_text(text, rel):
            chunk_id = Chunk.content_address(chunk_text)
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, source=rel))

    logger.info("Produced %d chunks from %d files", len(chunks), len(files))
    return chunks


def run_multihead(
    suite_path: Path,
    corpus_dir: Path,
    n_results: int = 5,
    seed: int = 42,
    output_dir: Path | None = None,
    use_gpu: bool = False,
    doc2query_model: str = "qwen3:4b",
    doc2query_api_url: str = "http://localhost:11434",
    doc2query_timeout_s: float = 30.0,
    doc2query_max_retries: int = 2,
    doc2query_n_questions: int = 3,
    custom_prompt: str = "",
    adaptive_n: bool = False,
    max_questions_per_chunk: int = 12,
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
    alpha: float = 0.5,
    cache_dir: Path | None = None,
) -> str:
    """Execute the multi-head benchmark harness and return comparison matrix JSON."""
    from .doc2query import Doc2QueryConfig, Doc2QueryGenerator
    from .embedding import EmbeddingConfig, HashEmbedder, TransformersEmbedder
    from .harness import run_multihead_benchmark
    from .heads.head_a import HeadA
    from .heads.head_b import HeadB
    from .heads.head_c import HeadC
    from .suite import BenchSuite

    suite = BenchSuite.from_yaml(
        suite_path,
        n_results_override=n_results,
        seed_override=seed,
    )

    # Chunk corpus
    chunks = _chunk_corpus(corpus_dir)
    if not chunks:
        raise RuntimeError(f"No indexable files found in {corpus_dir}")

    # Build embedder
    if use_gpu:
        embedder = TransformersEmbedder(EmbeddingConfig(
            model_name=embedding_model,
        ))
    else:
        embedder = HashEmbedder()

    # Build Doc2Query generator
    doc2query = Doc2QueryGenerator(Doc2QueryConfig(
        model=doc2query_model,
        api_url=doc2query_api_url,
        timeout_s=doc2query_timeout_s,
        max_retries=doc2query_max_retries,
        n_questions=doc2query_n_questions,
        custom_prompt=custom_prompt,
        adaptive_n=adaptive_n,
        max_questions_per_chunk=max_questions_per_chunk,
    ))

    cache_path = cache_dir / "augmentation_cache.json" if cache_dir else None

    # Build heads
    head_a = HeadA(chunks)
    head_b = HeadB(embedder=embedder, doc2query=doc2query, cache_path=cache_path)
    head_b.build_index(chunks)
    head_c = HeadC(head_a=head_a, head_b=head_b, alpha=alpha)

    # Sidecar path for DoD item 5
    sidecar_path = output_dir / "questions_sidecar.json" if output_dir else None

    return run_multihead_benchmark(
        suite=suite,
        chunks=chunks,
        head_a=head_a,
        head_b=head_b,
        head_c=head_c,
        corpus_dir=str(corpus_dir),
        n_results=n_results,
        sidecar_path=sidecar_path,
    )


# ── Prefill runner ───────────────────────────────────────────────────────────


def prefill_doc2query(
    corpus_dir: Path,
    doc2query_model: str = "qwen3:4b",
    doc2query_api_url: str = "http://localhost:11434",
    doc2query_timeout_s: float = 30.0,
    doc2query_max_retries: int = 2,
    doc2query_n_questions: int = 3,
    custom_prompt: str = "",
    adaptive_n: bool = False,
    max_questions_per_chunk: int = 12,
    cache_dir: Path | None = None,
) -> str:
    """Pre-populate Doc2Query augmentation cache without building the embedding index.

    Phase 1a of sequential g6.xlarge execution: call this with ollama running
    (qwen3:4b int8, ~2.5 GB VRAM), then stop ollama, then run
    ``multihead --use-gpu`` with the same ``--cache-dir``. ``build_index()``
    reads from the populated cache and makes zero Doc2Query API calls.

    Returns JSON string with prefill stats.
    """
    from .augmentation_cache import AugmentationCache
    from .doc2query import Doc2QueryConfig, Doc2QueryGenerator

    chunks = _chunk_corpus(corpus_dir)
    if not chunks:
        raise RuntimeError(f"No indexable files found in {corpus_dir}")

    config = Doc2QueryConfig(
        model=doc2query_model,
        api_url=doc2query_api_url,
        timeout_s=doc2query_timeout_s,
        max_retries=doc2query_max_retries,
        n_questions=doc2query_n_questions,
        custom_prompt=custom_prompt,
        adaptive_n=adaptive_n,
        max_questions_per_chunk=max_questions_per_chunk,
    )
    d2q = Doc2QueryGenerator(config)
    config_fp = config.config_fingerprint()

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "augmentation_cache.json" if cache_dir else None
    cache = AugmentationCache(cache_path) if cache_path is not None else None

    generated = skipped = failed = 0
    for chunk in chunks:
        cache_key = f"{chunk.chunk_id}:{config_fp}"
        if cache and not cache.should_process(cache_key):
            skipped += 1
            continue
        try:
            questions = d2q.generate(chunk.text)
            if questions:
                if cache:
                    cache.mark_success(cache_key, questions)
                generated += 1
            else:
                if cache:
                    cache.mark_failed(cache_key)
                failed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Doc2Query failed for chunk %s: %s", chunk.chunk_id, exc)
            if cache:
                cache.mark_failed(cache_key)
            failed += 1

    if cache:
        cache.save()

    stats = {
        "total_chunks": len(chunks),
        "generated": generated,
        "skipped_cached": skipped,
        "failed": failed,
        "cache_path": str(cache_path) if cache_path else None,
    }
    logger.info(
        "Prefill complete: %d/%d generated, %d skipped, %d failed",
        generated, len(chunks), skipped, failed,
    )
    return json.dumps(stats, indent=2, ensure_ascii=False)
