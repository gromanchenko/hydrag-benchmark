# Changelog

All notable changes to `hydrag-benchmark` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-04-06

### Added

- **`sys_sampler.py`**: Background thread-based system resource sampler (`PhaseSampler` context manager, 100 ms interval). Collects CPU, RAM, disk I/O, network I/O, and 14 GPU metrics (NVML) per benchmark phase.
- **GPU metrics via `nvidia-ml-py`**: `compute_util_pct`, `mem_bandwidth_util_pct`, `vram_used/free/total_mb`, `pcie_tx/rx_kb_s`, `pcie_replay_delta`, `power_w`, `temp_c`, `sm_clock_mhz`, `mem_clock_mhz`, `fan_speed_pct`, `ecc_sbe_delta`, `ecc_dbe_delta`, `nvlink_rx/tx_kb_s`.
- **`[metrics]` optional extra**: `pip install hydrag-benchmark[metrics]` pulls `psutil>=5.9` and `nvidia-ml-py>=11.0`.
- **`surreal-microbench` sys_metrics output**: Each result row now includes a `sys_metrics` block with per-phase `drop` / `insert` / `rebuild` / `query` snapshots containing CPU, RAM, disk, network, and GPU data.
- **`available_backends()`**: Helper returning `{"psutil": bool, "nvml_library": bool, "nvml": bool, "gpu_count": int}` for diagnostic inspection.
- **`scripts/surreal_microbench_launch.py`**: EC2 spot launcher that runs `surreal-microbench` on a GPU instance (`g4dn.xlarge`) with SurrealDB and uploads full `sys_metrics` JSON results to S3.

### Changed

- `beir_bench_launch.py` default wheel URLs updated to `hydrag-core==1.3.0` and `hydrag-benchmark==0.6.0`.
- NVML graceful degradation: `nvml_library` flag distinguishes "ImportError" (library absent) from "0 GPUs detected", giving precise CLI warnings.

### Fixed

- **SurrealDB v2.2.1 compat in `head_d_surreal.py`**: Updated to use `RecordID`-bound params for RELATE, health URL path stripping, and `vector::similarity::cosine()` ORDER BY fallback for the broken `<|n|>` ANN operator.
- **SurrealDB credentials passthrough**: CLI and BEIR runner paths now forward `surrealdb_username`/`surrealdb_password` so authenticated SurrealDB runs do not silently fall back to unauthenticated defaults (backport of 0.5.7 fix).

## [0.5.7] - 2026-03-26

### Fixed

- SurrealDB credentials are now passed through CLI and BEIR runner paths so authenticated SurrealDB runs do not silently fall back to unauthenticated defaults.

### Changed

- Package version bumped to `0.5.7`.

## [0.5.6] - 2026-03-25

### Added

- SurrealDB backend adapter support in BEIR benchmark workflows.

### Fixed

- Head E behavior fix in benchmark execution flow.

### Changed

- Package version bumped to `0.5.6`.

## [0.5.5] - 2026-03-21

### Fixed

- **Version sync**: `__version__` in `__init__.py` now matches `pyproject.toml` (was stuck at `0.5.2` while pyproject.toml was at `0.5.4`).

### Changed

- Package version bumped to `0.5.5`.

## [0.5.4] - 2026-03-19

### Added

- `--skip-indexing` CLI flag to reuse existing ChromaDB index without re-indexing (T-762).

### Changed

- Package version bumped to `0.5.4`.

## [0.5.3] - 2026-03-18

### Fixed

- **ZIP path traversal** (C1): `beir_loader.py` validates `namelist()` members for `..` and `/` prefixes; Python 3.12+ uses `filter="data"` on `extractall()`
- **Recall metric** (C2): `recall_at_k` returns `0.0` for empty `relevant_phrases` (was `1.0`, vacuously inflated)
- **Chunk overlap** (C3): `chunk_overlap` uses token set intersection instead of substring `in` matching (false-positived on partial tokens)
- **Head lifecycle leak** (C4): `beir_runner.py` wraps head lifecycle in `try/finally` to guarantee `head.close()` on exception
- **Head D private API** (H2): `head_d.py` rewritten to use `keyword_search()` public API + `text_to_id` map (was accessing `_conn` and `_escape_fts_query`)
- **Silent file errors** (H3): `runner.py` bare `except Exception: continue` now logs `logger.warning` with filename and exception
- **Head C fallback** (M4): Embedding-absent chunks scored at full `structural_norm` instead of `α * structural_norm`
- **Head A normalization** (M5): Raw scores normalized to `[0, 1]` via max-score division to prevent RRF fusion dominance

### Changed

- Package version bumped to `0.5.3`.

## [0.5.0] - 2026-03-18

### Changed

- **Doc2Query & AugmentationCache re-exported from hydrag-core** (T-746): `doc2query.py` and `augmentation_cache.py` are now thin re-export wrappers; canonical implementation lives in `hydrag.doc2query` (hydrag-core v1.2.0+). All existing imports remain backward-compatible.
- Minor version bump for hydrag-core v1.2.0 compatibility.
- Package version bumped to `0.5.0`.

## [0.3.0] - 2026-03-15

### Added

- `prefill` CLI command for Phase 1a Doc2Query cache population.
- `multihead` CLI command for A/B/C configuration harness execution.
- GPU optional extra (`hydrag-benchmark[gpu]`) for transformers-based embedding path.
- New benchmark suites: `k8s-kep.yaml` and `cpython-stdlib.yaml`.

### Changed

- Package version bumped to `0.3.0`.
- README updated for complete CLI argument, config, and artifact-path parity.

## [0.2.0] - 2026-03-15

### Added

- Strategy-aware dispatch: `similarity`, `hybrid`, `crag`, `hydrag` each route
  through appropriate hydrag-core retrieval heads.
- `_ChromaDBAdapter` implementing `VectorStoreAdapter` protocol for hydrag-core
  integration.
- `SUPPORTED_STRATEGIES` and `_STRATEGY_HEADS` constants in `runner.py`.
- 13 new strategy dispatch tests (`test_strategy.py`).
- CI job with pytest + smoke benchmark + recall gate (`recall_at_k >= 0.5`).

### Changed

- `_search_fn` replaced with strategy-aware dispatch (was direct ChromaDB query).

## [0.1.0] - 2026-03-14

### Added

- Initial release.
- `hydrag-bench` CLI with `run` and `list-suites` subcommands.
- Benchmark suite YAML format with versioned schema.
- Frozen v0.1 metrics: `recall_at_1`, `recall_at_k`, `mrr`, `chunk_overlap`,
  `latency_ms` (avg, p50, p95, p99).
- Versioned JSON output schema (`schema_version: "0.1"`).
- Synthetic smoke-test suite (`suites/synthetic-smoke.yaml`).
- PEP 561 `py.typed` marker for type-checked consumers.
- 30 unit tests covering runner, metrics, CLI, and suite parsing.
