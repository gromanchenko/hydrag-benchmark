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
    beir_p = sub.add_parser("beir", help="Run BEIR IR benchmark for FTS5 heads (D/E)")
    beir_p.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    beir_p.add_argument(
        "--heads", nargs="+", default=["head_d", "head_e"],
        help="Heads to benchmark (default: head_d head_e)",
    )
    beir_p.add_argument("--cache-dir", type=Path, default=None, help="BEIR dataset cache directory")
    beir_p.add_argument("--output-dir", type=Path, default=None, help="Directory to write JSON results")
    beir_p.add_argument("--max-queries", type=int, default=0, help="Limit queries (0 = all)")
    beir_p.add_argument("--ollama-model", default="qwen3:4b", help="Ollama model for Head E enrichment")
    beir_p.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API endpoint")

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
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
