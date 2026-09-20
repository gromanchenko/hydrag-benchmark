"""T-5052 A09 (B-02/B-03): max_queries must sample with a seeded RNG, not
take the first N in dict order, and must never reduce the corpus.

Previously: ``valid_qids = [qid for qid in queries if qid in qrels][:max_queries]``
always picked the same first N queries regardless of any seed -- the CLI's
--seed concept existed nowhere near this code path. Corpus indexing itself
was already unaffected by max_queries (confirmed here as a regression
guard), but the query-selection half of the subset story was not seeded.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hydrag_benchmark.beir_runner import run_beir_benchmark


@pytest.fixture()
def larger_beir_dataset(tmp_path: Path) -> Path:
    """A BEIR-format dataset with enough queries to distinguish a seeded
    sample from a first-N slice."""
    ds_dir = tmp_path / "larger-dataset"
    ds_dir.mkdir()

    corpus = [
        {"_id": str(i), "title": f"Doc {i}", "text": f"Document number {i} about topic {i}."}
        for i in range(1, 11)
    ]
    with open(ds_dir / "corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    queries = [{"_id": f"q{i:02d}", "text": f"topic {i}"} for i in range(1, 11)]
    with open(ds_dir / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    (ds_dir / "qrels").mkdir()
    with open(ds_dir / "qrels" / "test.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for i in range(1, 11):
            f.write(f"q{i:02d}\t{i}\t1\n")

    return ds_dir


def _run(dataset_dir: Path, tmp_path: Path, *, max_queries: int, seed: int, out_name: str) -> object:
    with patch("hydrag_benchmark.beir_runner.download_beir_dataset", return_value=dataset_dir):
        return run_beir_benchmark(
            dataset="scifact",
            heads=["fts5_baseline"],
            cache_dir=tmp_path,
            output_dir=tmp_path / out_name,
            max_queries=max_queries,
            seed=seed,
        )


def _query_ids(result: object) -> set[str]:
    head = next(hr for hr in result.heads if hr.head == "fts5_baseline")  # type: ignore[attr-defined]
    return {q.query_id for q in head.queries}


class TestSeededQuerySampling:
    def test_different_seeds_select_different_queries(
        self, larger_beir_dataset: Path, tmp_path: Path
    ) -> None:
        ids_a = _query_ids(_run(larger_beir_dataset, tmp_path, max_queries=3, seed=1, out_name="a"))
        ids_b = _query_ids(_run(larger_beir_dataset, tmp_path, max_queries=3, seed=2, out_name="b"))
        assert len(ids_a) == 3
        assert len(ids_b) == 3
        assert ids_a != ids_b, "different seeds must not always pick the same first N queries"

    def test_same_seed_is_reproducible(self, larger_beir_dataset: Path, tmp_path: Path) -> None:
        ids_a = _query_ids(_run(larger_beir_dataset, tmp_path, max_queries=4, seed=7, out_name="a"))
        ids_b = _query_ids(_run(larger_beir_dataset, tmp_path, max_queries=4, seed=7, out_name="b"))
        assert ids_a == ids_b

    def test_max_queries_does_not_reduce_corpus(
        self, larger_beir_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(larger_beir_dataset, tmp_path, max_queries=2, seed=1, out_name="c")
        head = next(hr for hr in result.heads if hr.head == "fts5_baseline")
        assert head.n_corpus == 10, "max_queries must only limit queries, never the indexed corpus"
