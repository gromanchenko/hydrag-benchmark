# hydrag-benchmark

Local-only RAG benchmarking CLI for retrieval quality and latency analysis.

## Installation

```bash
pip install hydrag-benchmark
```

Optional GPU path for multi-head dense embeddings:

```bash
pip install "hydrag-benchmark[gpu]"
```

## Included Suites

- `suites/synthetic-smoke.yaml`
- `suites/k8s-kep.yaml`
- `suites/cpython-stdlib.yaml`

## Quickstart

```bash
# List shipped suites
hydrag-bench list-suites --suite-dir ./suites

# Run classic strategy benchmark
hydrag-bench run suites/synthetic-smoke.yaml \
  --strategy hydrag \
  --corpus-dir ./my-codebase/src \
  --output-dir ./results

# Inspect output
python -m json.tool ./results/synthetic-smoke_hydrag.json
```

## Commands

```bash
hydrag-bench --help
hydrag-bench --version

# 1) Classic single-strategy benchmark
hydrag-bench run <suite.yaml> --strategy <similarity|hybrid|crag|hydrag> --corpus-dir <path> [options]

# 2) List suites
hydrag-bench list-suites --suite-dir <path>

# 3) Prefill Doc2Query cache (Phase 1a)
hydrag-bench prefill --corpus-dir <path> [options]

# 4) Multi-head harness benchmark (Heads A/B/C)
hydrag-bench multihead <suite.yaml> --corpus-dir <path> [options]
```

### `run` Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `suite` | yes | - | Path to benchmark suite YAML |
| `--strategy` | yes | - | One of `similarity`, `hybrid`, `crag`, `hydrag` |
| `--corpus-dir` | yes | - | Root directory of files to index |
| `--output-dir` | no | stdout | Directory to write `<suite>_<strategy>.json` |
| `--suite-dir` | no | - | Base dir for resolving relative `suite` path |
| `--n-results` | no | `5` | Top-k retrieval depth |
| `--seed` | no | `42` | Seed override |
| `--embedding-model` | no | `Alibaba-NLP/gte-Qwen2-7B-instruct` | Embedding model label passed to runner |
| `--db-path` | no | temp dir | ChromaDB persistence path |

### `list-suites` Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--suite-dir` | yes | - | Directory containing `.yaml` / `.yml` suites |

### `prefill` Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--corpus-dir` | yes | - | Root directory to chunk and process |
| `--doc2query-model` | no | `qwen3:4b` | Doc2Query model name |
| `--doc2query-api-url` | no | `http://localhost:11434` | Doc2Query API base URL |
| `--doc2query-timeout-s` | no | `30.0` | Request timeout seconds |
| `--doc2query-max-retries` | no | `2` | Retry attempts after first failure |
| `--doc2query-n-questions` | no | `3` | Synthetic questions per chunk |
| `--cache-dir` | no | in-memory only | Directory containing `augmentation_cache.json` |

### `multihead` Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `suite` | yes | - | Path to benchmark suite YAML |
| `--corpus-dir` | yes | - | Root directory of files to index |
| `--output-dir` | no | stdout | Directory to write `<suite>_multihead.json` and sidecar |
| `--suite-dir` | no | - | Base dir for resolving relative `suite` path |
| `--n-results` | no | `5` | Top-k retrieval depth |
| `--seed` | no | `42` | Seed override |
| `--use-gpu` | no | `false` | Use transformers embedder (requires `[gpu]`) |
| `--doc2query-model` | no | `qwen3:4b` | Doc2Query model name |
| `--doc2query-api-url` | no | `http://localhost:11434` | Doc2Query API base URL |
| `--doc2query-timeout-s` | no | `30.0` | Request timeout seconds |
| `--doc2query-max-retries` | no | `2` | Retry attempts after first failure |
| `--doc2query-n-questions` | no | `3` | Synthetic questions per chunk |
| `--embedding-model` | no | `Alibaba-NLP/gte-Qwen2-7B-instruct` | Dense embedding model name |
| `--alpha` | no | `0.5` | Head C rerank interpolation weight |
| `--cache-dir` | no | none | Directory for `augmentation_cache.json` persistence |

## Config Variables and Runtime Inputs

- `hydrag-benchmark` does not read `HYDRAG_BENCHMARK_*` environment variables.
- Operator-facing runtime configuration is via CLI flags and suite YAML fields.
- Suite-level fields consumed by code:
  - top-level: `name`, `version`, `seed`, `description`, `cases`
  - `environment`: `strategy`, `n_results`

## File Paths and Artifacts

| Path / Pattern | Producer | Meaning |
|----------------|----------|---------|
| `<output-dir>/<suite>_<strategy>.json` | `run` | Single-strategy result JSON (`schema_version: 0.1`) |
| `<output-dir>/<suite>_multihead.json` | `multihead` | Multi-head comparison matrix (`schema_version: 0.2`) |
| `<output-dir>/questions_sidecar.json` | `multihead` | Head B generated questions sidecar |
| `<cache-dir>/augmentation_cache.json` | `prefill` / `multihead` | 3-state Doc2Query cache shared across phases |
| `<db-path>` | `run` | ChromaDB persistent store location |

## Output Schemas

- `run` emits schema `0.1` with per-case and aggregate metrics.
- `multihead` emits schema `0.2` with 5 config groups:
  - `A-only`
  - `B-only`
  - `C-only`
  - `A+B`
  - `A+B+C`

## Frozen 0.1 Metrics

| Metric | Description |
|--------|-------------|
| `recall_at_1` | 1.0 when top result includes a relevant phrase |
| `recall_at_k` | Fraction of relevant phrases found in top-k |
| `mrr` | Mean Reciprocal Rank of first relevant result |
| `chunk_overlap` | Token overlap between retrieved chunks and relevant phrases |
| `latency_ms.avg` | Mean latency in milliseconds |
| `latency_ms.p50` | 50th percentile latency |
| `latency_ms.p95` | 95th percentile latency |
| `latency_ms.p99` | 99th percentile latency |

## Suite YAML Format

```yaml
name: my-benchmark
version: "1.0"
seed: 42
description: Description of the benchmark suite.

environment:
  strategy: hydrag
  n_results: 5

cases:
  - id: case-001
    query: "search query text"
    relevant_phrases:
      - "expected phrase in results"
      - "another expected phrase"
    tags: [optional, tags]
```

## Development

```bash
cd packages/hydrag-benchmark
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

Apache-2.0
