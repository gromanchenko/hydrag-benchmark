"""Head/backend matrix runner for corpus-local benchmarks.

Evaluates selected heads against selected storage backends on the same corpus
and query suite. This is used for scenarios like:
  - symbol_graph vs hydrag_full
  - sqlite vs surrealdb backend
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .heads.base import Chunk
from .heads.head_a import HeadA
from .heads.head_hydrag import HeadHydrag
from .metrics import chunk_overlap, latency_stats, mrr, recall_at_1, recall_at_k
from .suite import BenchSuite

SUPPORTED_MATRIX_HEADS: frozenset[str] = frozenset({"symbol_graph", "hydrag_full"})
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"sqlite", "surrealdb"})
SCHEMA_VERSION = "0.1-matrix"


@dataclass
class MatrixCaseResult:
    case_id: str
    query: str
    head: str
    backend: str
    recall_at_1: float
    recall_at_k: float
    mrr: float
    chunk_overlap: float
    latency_ms: float
    error: str | None = None


@dataclass
class MatrixComboResult:
    head: str
    backend: str
    total_cases: int
    recall_at_1: float
    recall_at_k: float
    mrr: float
    chunk_overlap: float
    latency_ms: dict[str, float]
    notes: list[str] = field(default_factory=lambda: [])


@dataclass
class MatrixRunResult:
    schema_version: str
    run_id: str
    timestamp: str
    tool_version: str
    suite_name: str
    corpus_dir: str
    heads: list[str]
    backends: list[str]
    n_results: int
    seed: int
    embedding_model: str
    inference_model: str
    cases: list[MatrixCaseResult]
    matrix: list[MatrixComboResult]


def _normalize_list(values: list[str], supported: frozenset[str], label: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = raw.strip().lower()
        if value not in supported:
            raise ValueError(
                f"Unsupported {label}: {raw!r}. Supported: {', '.join(sorted(supported))}"
            )
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


_CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".go", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".java", ".js", ".jsx", ".ts", ".tsx", ".rs", ".rb",
    ".scala", ".kt", ".swift", ".md", ".yaml", ".yml",
    ".toml", ".json", ".sh", ".bash", ".tf",
})


def _chunk_text(text: str, source: str, max_chars: int = 2000) -> list[tuple[str, str]]:
    """Split text into chunks by paragraph boundaries."""
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


def _chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    """Parse corpus files into content-addressed chunks."""
    files = [
        fp
        for fp in corpus_dir.rglob("*")
        if fp.is_file() and fp.suffix in _CODE_EXTENSIONS
    ]
    chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            continue
        rel = str(fp.relative_to(corpus_dir))
        for _, chunk_text in _chunk_text(text, rel):
            chunk_id = Chunk.content_address(chunk_text)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, source=rel))

    return chunks


def _to_texts(results: list[Any]) -> list[str]:
    out: list[str] = []
    for result in results:
        chunk = getattr(result, "chunk", None)
        if chunk is not None and hasattr(chunk, "text"):
            out.append(chunk.text)
            continue
        text = getattr(result, "text", None)
        if isinstance(text, str):
            out.append(text)
    return out


def _build_head(
    *,
    head_name: str,
    backend: str,
    chunks: list[Chunk],
    sqlite_db_path: Path,
    ollama_host: str,
    inference_model: str,
    embedding_model: str,
    surrealdb_url: str,
    surrealdb_namespace: str,
    surrealdb_database: str,
    surrealdb_username: str | None,
    surrealdb_password: str | None,
    surrealdb_token: str | None,
    surrealdb_embedding_dim: int,
) -> tuple[Any, list[str]]:
    notes: list[str] = []

    if head_name == "symbol_graph":
        if backend != "sqlite":
            notes.append("symbol_graph is backend-agnostic; result should match sqlite run")
        return HeadA(chunks), notes

    if head_name == "hydrag_full":
        head = HeadHydrag(
            db_path=sqlite_db_path,
            db_backend=backend,
            ollama_host=ollama_host,
            ollama_model=inference_model,
            surrealdb_url=surrealdb_url,
            surrealdb_namespace=surrealdb_namespace,
            surrealdb_database=surrealdb_database,
            surrealdb_username=surrealdb_username,
            surrealdb_password=surrealdb_password,
            surrealdb_token=surrealdb_token,
            surrealdb_embedding_model=embedding_model,
            surrealdb_embedding_dim=surrealdb_embedding_dim,
        )
        return head, notes

    raise ValueError(f"Unsupported head: {head_name!r}")


def _evaluate_combo(
    *,
    head: Any,
    head_name: str,
    backend: str,
    suite: BenchSuite,
    n_results: int,
) -> tuple[list[MatrixCaseResult], MatrixComboResult]:
    case_results: list[MatrixCaseResult] = []
    latencies: list[float] = []

    for case in suite.cases:
        t0 = time.monotonic()
        error: str | None = None
        retrieved: list[str] = []

        try:
            retrieved = _to_texts(head.retrieve(case.query, n_results=n_results))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        latencies.append(latency_ms)

        case_results.append(
            MatrixCaseResult(
                case_id=case.id,
                query=case.query,
                head=head_name,
                backend=backend,
                recall_at_1=recall_at_1(retrieved, case.relevant_phrases),
                recall_at_k=recall_at_k(retrieved, case.relevant_phrases),
                mrr=mrr(retrieved, case.relevant_phrases),
                chunk_overlap=chunk_overlap(retrieved, case.relevant_phrases),
                latency_ms=latency_ms,
                error=error,
            )
        )

    total = len(case_results)
    combo = MatrixComboResult(
        head=head_name,
        backend=backend,
        total_cases=total,
        recall_at_1=round(sum(c.recall_at_1 for c in case_results) / total, 3) if total else 0.0,
        recall_at_k=round(sum(c.recall_at_k for c in case_results) / total, 3) if total else 0.0,
        mrr=round(sum(c.mrr for c in case_results) / total, 3) if total else 0.0,
        chunk_overlap=round(sum(c.chunk_overlap for c in case_results) / total, 3) if total else 0.0,
        latency_ms=latency_stats(latencies),
        notes=[],
    )
    return case_results, combo


def run_head_backend_matrix(
    *,
    suite_path: Path,
    corpus_dir: Path,
    heads: list[str],
    backends: list[str],
    n_results: int = 5,
    seed: int = 42,
    ollama_host: str = "http://localhost:11434",
    inference_model: str = "qwen3:4b",
    embedding_model: str = "nomic-embed-text",
    surrealdb_url: str = "ws://localhost:8000/rpc",
    surrealdb_namespace: str = "hydrag_benchmark",
    surrealdb_database_prefix: str = "httpd",
    surrealdb_username: str | None = None,
    surrealdb_password: str | None = None,
    surrealdb_token: str | None = None,
    surrealdb_embedding_dim: int = 768,
    working_dir: Path | None = None,
) -> str:
    """Run matrix benchmark across heads and storage backends.

    Defaults are tuned for g6.2xlarge (L4):
      - inference model: qwen3:4b
      - embedding model: nomic-embed-text
    """
    suite = BenchSuite.from_yaml(
        suite_path,
        n_results_override=n_results,
        seed_override=seed,
    )
    normalized_heads = _normalize_list(heads, SUPPORTED_MATRIX_HEADS, "head")
    normalized_backends = _normalize_list(backends, SUPPORTED_BACKENDS, "backend")

    chunks = _chunk_corpus(corpus_dir)
    if not chunks:
        raise RuntimeError(f"No indexable files found in {corpus_dir}")

    run_token = uuid.uuid4().hex[:8]
    base_work_dir = (working_dir or (Path("/tmp") / f"hydrag-matrix-{run_token}")).resolve()
    base_work_dir.mkdir(parents=True, exist_ok=True)

    all_cases: list[MatrixCaseResult] = []
    matrix: list[MatrixComboResult] = []

    for backend in normalized_backends:
        for head_name in normalized_heads:
            sqlite_db_path = base_work_dir / f"{backend}-{head_name}.sqlite3"
            surreal_db = f"{surrealdb_database_prefix}_{run_token}_{head_name}_{backend}".replace("-", "_")

            head, notes = _build_head(
                head_name=head_name,
                backend=backend,
                chunks=chunks,
                sqlite_db_path=sqlite_db_path,
                ollama_host=ollama_host,
                inference_model=inference_model,
                embedding_model=embedding_model,
                surrealdb_url=surrealdb_url,
                surrealdb_namespace=surrealdb_namespace,
                surrealdb_database=surreal_db,
                surrealdb_username=surrealdb_username,
                surrealdb_password=surrealdb_password,
                surrealdb_token=surrealdb_token,
                surrealdb_embedding_dim=surrealdb_embedding_dim,
            )

            closeable = hasattr(head, "close")
            try:
                if hasattr(head, "build_index"):
                    head.build_index(chunks)
                case_results, combo = _evaluate_combo(
                    head=head,
                    head_name=head_name,
                    backend=backend,
                    suite=suite,
                    n_results=n_results,
                )
                combo.notes.extend(notes)
                all_cases.extend(case_results)
                matrix.append(combo)
            finally:
                if closeable:
                    head.close()

    result = MatrixRunResult(
        schema_version=SCHEMA_VERSION,
        run_id=f"matrix-{run_token}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool_version=__version__,
        suite_name=suite.name,
        corpus_dir=str(corpus_dir),
        heads=normalized_heads,
        backends=normalized_backends,
        n_results=n_results,
        seed=seed,
        embedding_model=embedding_model,
        inference_model=inference_model,
        cases=all_cases,
        matrix=matrix,
    )
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)
