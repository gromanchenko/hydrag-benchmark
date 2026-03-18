"""BEIR dataset loader — downloads and caches standard IR evaluation datasets.

Supports BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/test.tsv).
Downloads from HuggingFace BEIR mirror if not cached locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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

# SHA-256 checksums for BEIR dataset archives (S-002: verify external downloads)
# Note: BEIR datasets require network access — opt-in behavior (S-003 acknowledged).
# Computed from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip
BEIR_SHA256: dict[str, str] = {
    "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "nfcorpus": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    "arguana": "cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
    "scidocs": "96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1",
    "fiqa": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
}


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

    # SHA-256 verification (S-002)
    expected_hash = BEIR_SHA256.get(dataset)
    if expected_hash:
        actual_hash = _sha256_file(archive_path)
        if actual_hash != expected_hash:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {dataset}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        logger.info("SHA-256 verified for %s", dataset)
    else:
        logger.warning(
            "No SHA-256 hash registered for BEIR dataset %s — skipping verification",
            dataset,
        )

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


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
