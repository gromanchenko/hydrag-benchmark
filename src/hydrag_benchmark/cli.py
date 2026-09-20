"""hydrag-benchmark CLI entry point — argparse, no external deps beyond stdlib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydrag-bench",
        description="Local-only RAG benchmarking CLI — measures recall, MRR, chunk overlap, and latency.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # ── run ───────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Execute a benchmark suite")
    run_p.add_argument("suite", type=Path, help="Path to benchmark suite YAML file")
    run_p.add_argument("--strategy", required=True, help="Retrieval strategy name (e.g. hydrag, hybrid, crag)")
    run_p.add_argument("--corpus-dir", type=Path, required=True, help="Root directory of the corpus to index")
    run_p.add_argument("--output-dir", type=Path, default=None, help="Directory to write JSON results (default: stdout)")
    run_p.add_argument("--suite-dir", type=Path, default=None, help="Base directory for resolving relative suite paths")
    run_p.add_argument("--n-results", type=int, default=5, help="Top-k results to retrieve per query (default: 5)")
    run_p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    run_p.add_argument(
        "--embedding-model",
        default="Alibaba-NLP/gte-Qwen2-7B-instruct",
        help="Embedding model name (default: Alibaba-NLP/gte-Qwen2-7B-instruct)",
    )
    run_p.add_argument("--db-path", type=Path, default=None, help="ChromaDB persistence path (default: temp dir)")

    # ── list-suites ───────────────────────────────────────────────
    list_p = sub.add_parser("list-suites", help="List available benchmark suite YAML files")
    list_p.add_argument("--suite-dir", type=Path, required=True, help="Directory containing suite YAML files")

    # ── prefill ───────────────────────────────────────────────────
    pf_p = sub.add_parser(
        "prefill",
        help="Pre-populate Doc2Query augmentation cache (Phase 1a sequential execution)",
    )
    pf_p.add_argument(
        "--corpus-dir", type=Path, required=True,
        help="Root directory of the corpus to chunk and process",
    )
    pf_p.add_argument(
        "--doc2query-model", default="qwen3:4b",
        help="Doc2Query LLM model via ollama (default: qwen3:4b)",
    )
    pf_p.add_argument(
        "--doc2query-api-url", default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    pf_p.add_argument(
        "--doc2query-timeout-s", type=float, default=30.0,
        help="Doc2Query request timeout in seconds (default: 30)",
    )
    pf_p.add_argument(
        "--doc2query-max-retries", type=int, default=2,
        help="Doc2Query retry count after first failure (default: 2)",
    )
    pf_p.add_argument(
        "--doc2query-n-questions", type=int, default=3,
        help="Synthetic questions per chunk (default: 3)",
    )
    pf_p.add_argument(
        "--custom-prompt", default="",
        help="Custom context prepended to the Doc2Query prompt template",
    )
    pf_p.add_argument(
        "--adaptive-n", action="store_true", default=False,
        help="Adapt question count to chunk token length (RFC §2.3)",
    )
    pf_p.add_argument(
        "--max-questions-per-chunk", type=int, default=12,
        help="Cap on questions per chunk when adaptive_n is enabled (default: 12)",
    )
    pf_p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory for augmentation cache (required for Phase 1b to read from)",
    )

    # ── multihead ─────────────────────────────────────────────────
    mh_p = sub.add_parser("multihead", help="Run multi-head retrieval benchmark (Heads A/B/C)")
    mh_p.add_argument("suite", type=Path, help="Path to benchmark suite YAML file")
    mh_p.add_argument("--corpus-dir", type=Path, required=True, help="Root directory of the corpus to index")
    mh_p.add_argument("--output-dir", type=Path, default=None, help="Directory to write JSON results and question sidecar")
    mh_p.add_argument("--suite-dir", type=Path, default=None, help="Base directory for resolving relative suite paths")
    mh_p.add_argument("--n-results", type=int, default=5, help="Top-k results per query (default: 5)")
    mh_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    mh_p.add_argument("--use-gpu", action="store_true", help="Use GPU-accelerated transformers embedder (requires [gpu] extra)")
    mh_p.add_argument("--doc2query-model", default="qwen3:4b", help="Doc2Query LLM model (default: qwen3:4b)")
    mh_p.add_argument("--doc2query-api-url", default="http://localhost:11434", help="Doc2Query LLM API URL (default: ollama localhost)")
    mh_p.add_argument("--doc2query-timeout-s", type=float, default=30.0, help="Doc2Query request timeout in seconds (default: 30)")
    mh_p.add_argument("--doc2query-max-retries", type=int, default=2, help="Doc2Query retry count after first failure (default: 2)")
    mh_p.add_argument("--doc2query-n-questions", type=int, default=3, help="Synthetic questions per chunk (default: 3)")
    mh_p.add_argument("--custom-prompt", default="", help="Custom context prepended to the Doc2Query prompt template")
    mh_p.add_argument("--adaptive-n", action="store_true", default=False, help="Adapt question count to chunk token length (RFC §2.3)")
    mh_p.add_argument("--max-questions-per-chunk", type=int, default=12, help="Cap on questions per chunk when adaptive_n is enabled (default: 12)")
    mh_p.add_argument(
        "--embedding-model",
        default="Alibaba-NLP/gte-Qwen2-7B-instruct",
        help="Dense embedding model (default: gte-Qwen2-7B-instruct)",
    )
    mh_p.add_argument("--alpha", type=float, default=0.5, help="Head C rerank weight α (default: 0.5)")
    mh_p.add_argument("--cache-dir", type=Path, default=None, help="Directory for augmentation cache persistence")

    # ── beir ────────────────────────────────────────────────────────
    beir_p = sub.add_parser("beir", help="Run BEIR IR benchmark for canonical heads (legacy aliases accepted)")
    beir_p.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    beir_p.add_argument(
        "--heads", nargs="+", default=["fts5_baseline", "fts5_enriched"],
        help=(
            "Heads to benchmark (default: fts5_baseline fts5_enriched). "
            "Aliases head_a/head_b/head_c/head_d/head_e/head_hydrag are accepted."
        ),
    )
    beir_p.add_argument("--cache-dir", type=Path, default=None, help="BEIR dataset cache directory")
    beir_p.add_argument("--output-dir", type=Path, default=None, help="Directory to write JSON results")
    beir_p.add_argument("--max-queries", type=int, default=0, help="Limit queries (0 = all)")
    beir_p.add_argument("--seed", type=int, default=42, help="Random seed for --max-queries sampling (default: 42)")
    beir_p.add_argument("--ollama-model", default="qwen3:4b", help="Ollama model for Head E enrichment")
    beir_p.add_argument(
        "--max-corpus-for-enrichment",
        type=int,
        default=0,
        metavar="N",
        help=(
            "T-975: skip fts5_enriched when corpus > N docs (0 = use env var or "
            "module default 500K). Overrides HYDRAG_MAX_CORPUS_FOR_ENRICHMENT."
        ),
    )
    beir_p.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API endpoint")
    beir_p.add_argument(
        "--embedding-model",
        default="Alibaba-NLP/gte-Qwen2-7B-instruct",
        help="Dense embedding model for Head B/C (default: gte-Qwen2-7B-instruct)",
    )
    beir_p.add_argument("--use-gpu", action="store_true", help="Use GPU-accelerated TransformersEmbedder for Head B/C")
    beir_p.add_argument("--doc2query-model", default="qwen3:4b", help="Doc2Query LLM model for Head B (default: qwen3:4b)")
    beir_p.add_argument("--doc2query-api-url", default="http://localhost:11434", help="Doc2Query LLM API URL")
    # T-964: multi-backend comparison params
    beir_p.add_argument(
        "--surrealdb-url",
        default="ws://localhost:8000/rpc",
        help="SurrealDB RPC WebSocket URL for surreal_fts head (default: ws://localhost:8000/rpc)",
    )
    beir_p.add_argument(
        "--surrealdb-namespace",
        default="hydrag_beir",
        help="SurrealDB namespace for surreal_fts head (default: hydrag_beir)",
    )
    beir_p.add_argument(
        "--surrealdb-username",
        default=None,
        help="SurrealDB root username for surreal_fts head (default: None)",
    )
    beir_p.add_argument(
        "--surrealdb-password",
        default=None,
        help="SurrealDB root password for surreal_fts head (default: None)",
    )
    beir_p.add_argument(
        "--ollama-embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model for chroma_vector head (default: nomic-embed-text)",
    )
    # T-977: pre-seeded DB snapshot upload
    beir_p.add_argument(
        "--save-snapshots",
        action="store_true",
        help="T-977: after indexing, upload a gzip'd .db artifact + manifest.json to S3",
    )
    beir_p.add_argument(
        "--snapshot-bucket",
        default="lp2-artifacts",
        help="S3 bucket for snapshot artifacts (default: lp2-artifacts)",
    )
    beir_p.add_argument(
        "--load-snapshot-dir",
        type=Path,
        default=None,
        help="T-977: local directory with pre-seeded .db.gz snapshots (skips indexing)",
    )

    # ── head-matrix ───────────────────────────────────────────────
    matrix_p = sub.add_parser(
        "head-matrix",
        help="Run head/backend matrix benchmark on a local corpus",
    )
    matrix_p.add_argument("suite", type=Path, help="Path to benchmark suite YAML file")
    matrix_p.add_argument("--corpus-dir", type=Path, required=True, help="Root directory of the corpus to index")
    matrix_p.add_argument(
        "--heads",
        nargs="+",
        default=["symbol_graph", "hydrag_full"],
        help="Heads to benchmark (default: symbol_graph hydrag_full)",
    )
    matrix_p.add_argument(
        "--backends",
        nargs="+",
        default=["sqlite", "surrealdb"],
        help="Storage backends to benchmark (default: sqlite surrealdb)",
    )
    matrix_p.add_argument("--output-dir", type=Path, default=None, help="Directory to write JSON results")
    matrix_p.add_argument("--n-results", type=int, default=5, help="Top-k results to retrieve per query (default: 5)")
    matrix_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    matrix_p.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API endpoint")
    matrix_p.add_argument(
        "--inference-model",
        default="qwen3:4b",
        help="HydRAG inference model (default: qwen3:4b; tuned for g6.2xlarge)",
    )
    matrix_p.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Embedding model for SurrealDB vector index (default: nomic-embed-text)",
    )
    matrix_p.add_argument("--surrealdb-url", default="ws://localhost:8000/rpc", help="SurrealDB RPC URL")
    matrix_p.add_argument("--surrealdb-namespace", default="hydrag_benchmark", help="SurrealDB namespace")
    matrix_p.add_argument("--surrealdb-username", default=None, help="SurrealDB username (optional)")
    matrix_p.add_argument("--surrealdb-password", default=None, help="SurrealDB password (optional)")
    matrix_p.add_argument("--surrealdb-token", default=None, help="SurrealDB auth token (optional)")
    matrix_p.add_argument(
        "--surrealdb-database-prefix",
        default="httpd",
        help="Prefix for generated SurrealDB database names (default: httpd)",
    )
    matrix_p.add_argument(
        "--surrealdb-embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension for SurrealDB vectors (default: 768)",
    )
    matrix_p.add_argument(
        "--working-dir",
        type=Path,
        default=None,
        help="Working directory for SQLite DB files and temp artifacts",
    )

    # ── surreal-microbench ────────────────────────────────────────
    smb_p = sub.add_parser(
        "surreal-microbench",
        help="Raw SurrealDB INSERT/FTS throughput microbenchmark",
    )
    smb_p.add_argument(
        "--surrealdb-url", default="ws://localhost:8000/rpc",
        help="SurrealDB RPC WebSocket URL (default: ws://localhost:8000/rpc)",
    )
    smb_p.add_argument("--surrealdb-namespace", default="hydrag_bench", help="SurrealDB namespace")
    smb_p.add_argument("--surrealdb-username", default=None, help="SurrealDB username")
    smb_p.add_argument("--surrealdb-password", default=None, help="SurrealDB password")
    smb_p.add_argument(
        "--doc-count", type=int, default=50_000,
        help="Number of synthetic docs to insert (default: 50000)",
    )
    smb_p.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[100, 500, 2000, 5000],
        help="Batch sizes to benchmark (default: 100 500 2000 5000)",
    )
    smb_p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write JSON results",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "list-suites":
        return _cmd_list_suites(args)

    if args.command == "run":
        return _cmd_run(args)

    if args.command == "prefill":
        return _cmd_prefill(args)

    if args.command == "multihead":
        return _cmd_multihead(args)

    if args.command == "beir":
        return _cmd_beir(args)

    if args.command == "head-matrix":
        return _cmd_head_matrix(args)

    if args.command == "surreal-microbench":
        return _cmd_surreal_microbench(args)

    parser.print_help()
    return 1


def _cmd_list_suites(args: argparse.Namespace) -> int:
    suite_dir: Path = args.suite_dir
    if not suite_dir.is_dir():
        print(f"Error: suite directory not found: {suite_dir}", file=sys.stderr)
        return 1
    yamls = sorted(suite_dir.glob("*.yaml")) + sorted(suite_dir.glob("*.yml"))
    if not yamls:
        print(f"No suite YAML files found in {suite_dir}")
        return 0
    for p in yamls:
        print(p.name)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_benchmark

    suite_path: Path = args.suite
    if args.suite_dir and not suite_path.is_absolute():
        suite_path = args.suite_dir / suite_path

    if not suite_path.exists():
        print(f"Error: suite file not found: {suite_path}", file=sys.stderr)
        return 1

    corpus_dir: Path = args.corpus_dir
    if not corpus_dir.is_dir():
        print(f"Error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    result_json = run_benchmark(
        suite_path=suite_path,
        strategy=args.strategy,
        corpus_dir=corpus_dir,
        n_results=args.n_results,
        seed=args.seed,
        embedding_model=args.embedding_model,
        db_path=args.db_path,
    )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_file = args.output_dir / f"{suite_path.stem}_{args.strategy}.json"
        out_file.write_text(result_json, encoding="utf-8")
        print(f"Results written to {out_file}")
    else:
        print(result_json)

    return 0


def _cmd_prefill(args: argparse.Namespace) -> int:
    from .runner import prefill_doc2query

    corpus_dir: Path = args.corpus_dir
    if not corpus_dir.is_dir():
        print(f"Error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    result = prefill_doc2query(
        corpus_dir=corpus_dir,
        doc2query_model=args.doc2query_model,
        doc2query_api_url=args.doc2query_api_url,
        doc2query_timeout_s=args.doc2query_timeout_s,
        doc2query_max_retries=args.doc2query_max_retries,
        doc2query_n_questions=args.doc2query_n_questions,
        custom_prompt=args.custom_prompt,
        adaptive_n=args.adaptive_n,
        max_questions_per_chunk=args.max_questions_per_chunk,
        cache_dir=args.cache_dir,
    )
    print(result)
    return 0


def _cmd_multihead(args: argparse.Namespace) -> int:
    from .runner import run_multihead

    suite_path: Path = args.suite
    if args.suite_dir and not suite_path.is_absolute():
        suite_path = args.suite_dir / suite_path

    if not suite_path.exists():
        print(f"Error: suite file not found: {suite_path}", file=sys.stderr)
        return 1

    corpus_dir: Path = args.corpus_dir
    if not corpus_dir.is_dir():
        print(f"Error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    result_json = run_multihead(
        suite_path=suite_path,
        corpus_dir=corpus_dir,
        n_results=args.n_results,
        seed=args.seed,
        output_dir=args.output_dir,
        use_gpu=args.use_gpu,
        doc2query_model=args.doc2query_model,
        doc2query_api_url=args.doc2query_api_url,
        doc2query_timeout_s=args.doc2query_timeout_s,
        doc2query_max_retries=args.doc2query_max_retries,
        doc2query_n_questions=args.doc2query_n_questions,
        custom_prompt=args.custom_prompt,
        adaptive_n=args.adaptive_n,
        max_questions_per_chunk=args.max_questions_per_chunk,
        embedding_model=args.embedding_model,
        alpha=args.alpha,
        cache_dir=args.cache_dir,
    )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_file = args.output_dir / f"{suite_path.stem}_multihead.json"
        out_file.write_text(result_json, encoding="utf-8")
        print(f"Results written to {out_file}")
    else:
        print(result_json)

    return 0


def _cmd_beir(args: argparse.Namespace) -> int:
    import logging

    from .beir_runner import run_beir_benchmark

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    run_beir_benchmark(
        dataset=args.dataset,
        heads=args.heads,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        seed=args.seed,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        embedding_model=args.embedding_model,
        use_gpu=args.use_gpu,
        doc2query_model=args.doc2query_model,
        doc2query_api_url=args.doc2query_api_url,
        surrealdb_url=args.surrealdb_url,
        surrealdb_namespace=args.surrealdb_namespace,
        surrealdb_username=args.surrealdb_username,
        surrealdb_password=args.surrealdb_password,
        ollama_embedding_model=args.ollama_embedding_model,
        max_enrich_corpus=args.max_corpus_for_enrichment,
        save_snapshots=args.save_snapshots,
        snapshot_bucket=args.snapshot_bucket,
        load_snapshot_dir=args.load_snapshot_dir,
    )
    return 0


def _cmd_head_matrix(args: argparse.Namespace) -> int:
    from .head_matrix_runner import run_head_backend_matrix

    suite_path: Path = args.suite
    if not suite_path.exists():
        print(f"Error: suite file not found: {suite_path}", file=sys.stderr)
        return 1

    corpus_dir: Path = args.corpus_dir
    if not corpus_dir.is_dir():
        print(f"Error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    result_json = run_head_backend_matrix(
        suite_path=suite_path,
        corpus_dir=corpus_dir,
        heads=args.heads,
        backends=args.backends,
        n_results=args.n_results,
        seed=args.seed,
        ollama_host=args.ollama_host,
        inference_model=args.inference_model,
        embedding_model=args.embedding_model,
        surrealdb_url=args.surrealdb_url,
        surrealdb_namespace=args.surrealdb_namespace,
        surrealdb_username=args.surrealdb_username,
        surrealdb_password=args.surrealdb_password,
        surrealdb_token=args.surrealdb_token,
        surrealdb_database_prefix=args.surrealdb_database_prefix,
        surrealdb_embedding_dim=args.surrealdb_embedding_dim,
        working_dir=args.working_dir,
    )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_file = args.output_dir / f"{suite_path.stem}_head_matrix.json"
        out_file.write_text(result_json, encoding="utf-8")
        print(f"Results written to {out_file}")
    else:
        print(result_json)

    return 0


def _cmd_surreal_microbench(args: argparse.Namespace) -> int:
    import hashlib
    import json
    import logging
    import time
    import uuid

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    log = logging.getLogger("surreal-microbench")

    from hydrag import IndexedChunk
    from hydrag.surreal_adapter import SurrealDBAdapter
    from .sys_sampler import PhaseSampler, available_backends

    doc_count: int = args.doc_count
    batch_sizes: list[int] = args.batch_sizes

    backends = available_backends()
    if not backends["psutil"]:
        log.warning("psutil not installed — CPU/RAM/disk/network metrics disabled. "
                    "Install with: pip install hydrag-benchmark[metrics]")
    if not backends["nvml"]:
        if not backends["nvml_library"]:
            log.warning("nvidia-ml-py not installed — GPU metrics disabled. "
                        "Install with: pip install hydrag-benchmark[metrics]")
        else:
            log.info("NVML library loaded but no NVIDIA GPU detected — GPU metrics will be null")

    # Generate synthetic docs once
    log.info("generating %d synthetic docs", doc_count)
    chunks: list[IndexedChunk] = []
    for i in range(doc_count):
        text = f"Synthetic document number {i} with varied content for FTS indexing benchmark. " * 3
        cid = f"bench_{i:08d}"
        chunks.append(IndexedChunk(
            chunk_id=cid,
            source=f"synth/{cid}",
            title=f"Doc {i}",
            raw_content=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest()[:32],
        ))

    results: list[dict] = []
    _dummy_embed: list[float] = [0.0]

    for bs in batch_sizes:
        db_name = f"microbench_{uuid.uuid4().hex[:8]}"
        log.info("=== batch_size=%d, db=%s, docs=%d ===", bs, db_name, doc_count)

        adapter = SurrealDBAdapter(
            url=args.surrealdb_url,
            embedding_dim=1,
            embed_fn=lambda t: _dummy_embed,  # noqa: ARG005
            namespace=args.surrealdb_namespace,
            database=db_name,
            timeout=300,
            auto_schema=True,
            username=args.surrealdb_username,
            password=args.surrealdb_password,
            allow_insecure_auth=True,
            batch_size=bs,
            assume_fresh=True,
            deferred_index=False,  # manual phase control for accurate timing
            fts_fields=["raw_content"],
        )
        adapter._connect()

        # Phase 0: drop ALL indexes (FTS + HNSW + UNIQUE) — fresh DB, single writer
        with PhaseSampler("drop") as _drop_sampler:
            adapter._bridge.run(adapter._async_drop_indexes(include_unique=True), timeout=60)
        _drop_metrics = _drop_sampler.result()

        # Phase 1: INSERT only — no indexes active; true raw-insert ceiling
        with PhaseSampler("insert") as _ins_sampler:
            t_insert = time.monotonic()
            created = adapter.index_documents(chunks, embeddings=None)
            insert_elapsed = time.monotonic() - t_insert
        _ins_metrics = _ins_sampler.result()
        insert_rate = created / insert_elapsed if insert_elapsed > 0 else 0

        # Phase 2: rebuild ALL indexes (FTS + HNSW + UNIQUE)
        with PhaseSampler("rebuild") as _rebuild_sampler:
            t_rebuild = time.monotonic()
            adapter._bridge.run(adapter._async_rebuild_indexes(created, include_unique=True), timeout=3600)
            rebuild_elapsed = time.monotonic() - t_rebuild
        _rebuild_metrics = _rebuild_sampler.result()

        # Phase 3: query latency (indexes fully built)
        query_latencies: list[float] = []
        test_queries = ["synthetic document", "varied content", "FTS indexing benchmark"]
        with PhaseSampler("query") as _query_sampler:
            for q in test_queries:
                qt0 = time.monotonic()
                adapter.keyword_search(q, n_results=10)
                query_latencies.append(time.monotonic() - qt0)
        _query_metrics = _query_sampler.result()

        avg_query_ms = sum(query_latencies) / len(query_latencies) * 1000

        row = {
            "batch_size": bs,
            "doc_count": doc_count,
            "created": created,
            "insert_elapsed_s": round(insert_elapsed, 2),
            "insert_docs_per_sec": round(insert_rate, 1),
            "rebuild_elapsed_s": round(rebuild_elapsed, 2),
            "total_ingest_s": round(insert_elapsed + rebuild_elapsed, 2),
            "avg_query_latency_ms": round(avg_query_ms, 1),
            "sys_metrics": {
                "drop": _drop_metrics.to_dict(),
                "insert": _ins_metrics.to_dict(),
                "rebuild": _rebuild_metrics.to_dict(),
                "query": _query_metrics.to_dict(),
            },
        }
        results.append(row)
        log.info(
            "batch_size=%d: %d docs — insert=%.1fs (%.0f docs/sec), "
            "rebuild=%.1fs, query=%.1fms",
            bs, created, insert_elapsed, insert_rate,
            rebuild_elapsed, avg_query_ms,
        )

        # Cleanup: remove benchmark database (best-effort)
        try:
            adapter._query(f"REMOVE DATABASE IF EXISTS {db_name}")
        except Exception:
            log.debug("cleanup of %s skipped (expected if still connected)", db_name)

        try:
            adapter.close()
        except Exception:
            pass

    output = json.dumps({"microbench_results": results}, indent=2)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_file = args.output_dir / "surreal_microbench.json"
        out_file.write_text(output, encoding="utf-8")
        log.info("results written to %s", out_file)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
