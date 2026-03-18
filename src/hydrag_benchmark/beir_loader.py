"""BEIR dataset loader — downloads and caches standard IR evaluation datasets.

Supports BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/test.tsv).
Downloads from HuggingFace BEIR mirror if not cached locally.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger("hydrag_benchmark.beir_loader")

BEIR_HF_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

# Datasets that are small enough for local benchmarking (<100 MB compressed)
BEIR_SMALL_DATASETS = frozenset({
    "scifact",
    "nfcorpus",
    "arguana",
    "scidocs",
    "fiqa",
})

BEIR_MEDIUM_DATASETS = frozenset({
    "trec-covid",
    "webis-touche2020",
    "dbpedia-entity",
    "fever",
    "climate-fever",
    "hotpotqa",
    "nq",
})

ALL_BEIR_DATASETS = BEIR_SMALL_DATASETS | BEIR_MEDIUM_DATASETS


def default_cache_dir() -> Path:
    """Default BEIR dataset cache directory."""
    return Path(os.environ.get("BEIR_CACHE_DIR", Path.home() / ".cache" / "beir"))


def download_beir_dataset(
    dataset: str,
    cache_dir: Path | None = None,
) -> Path:
    """Download a BEIR dataset to cache. Returns path to dataset directory.

    Skips download if dataset already exists in cache.
    """
    if dataset not in ALL_BEIR_DATASETS:
        raise ValueError(
            f"Unknown BEIR dataset {dataset!r}. "
            f"Available: {', '.join(sorted(ALL_BEIR_DATASETS))}"
        )

    cache = cache_dir or default_cache_dir()
    dataset_dir = cache / dataset

    if (dataset_dir / "corpus.jsonl").exists():
        logger.info("BEIR %s already cached at %s", dataset, dataset_dir)
        return dataset_dir

    cache.mkdir(parents=True, exist_ok=True)
    url = f"{BEIR_HF_BASE}/{dataset}.zip"
    archive_path = cache / f"{dataset}.zip"

    logger.info("Downloading BEIR %s from %s", dataset, url)
    _download_with_progress(url, archive_path)

    # Extract — BEIR datasets come as zip files
    logger.info("Extracting %s", archive_path)
    import zipfile

    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache)

    # Verify extraction
    if not (dataset_dir / "corpus.jsonl").exists():
        raise RuntimeError(
            f"Extraction failed: {dataset_dir / 'corpus.jsonl'} not found"
        )

    # Clean up archive
    archive_path.unlink(missing_ok=True)
    logger.info("BEIR %s ready at %s", dataset, dataset_dir)
    return dataset_dir


def _download_with_progress(url: str, dest: Path) -> None:
    """Download URL to file with progress logging."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:  # noqa: S310 — trusted BEIR URL
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    if pct % 20 == 0:
                        logger.info("  %d%% (%d / %d bytes)", pct, downloaded, total)


def load_beir_corpus(dataset_dir: Path) -> dict[str, dict[str, str]]:
    """Load corpus.jsonl -> {doc_id: {"title": ..., "text": ...}}"""
    corpus: dict[str, dict[str, str]] = {}
    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[str(obj["_id"])] = {
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
            }
    return corpus


def load_beir_queries(dataset_dir: Path) -> dict[str, str]:
    """Load queries.jsonl -> {query_id: query_text}"""
    queries: dict[str, str] = {}
    with open(dataset_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[str(obj["_id"])] = obj["text"]
    return queries


def load_beir_qrels(
    dataset_dir: Path,
    split: str = "test",
) -> dict[str, dict[str, int]]:
    """Load qrels/test.tsv -> {query_id: {doc_id: relevance_score}}"""
    qrels: dict[str, dict[str, int]] = {}
    qrel_path = dataset_dir / "qrels" / f"{split}.tsv"
    with open(qrel_path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    return qrels
