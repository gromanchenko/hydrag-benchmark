"""T-5060: importing hydrag_benchmark.beir_runner must not require the
optional ``surrealdb`` package.

hydrag-benchmark's own pyproject.toml only declares
``hydrag-core[chromadb]`` as a dependency, never ``[surrealdb]``. A
module-level ``from .heads.head_d_surreal import HeadDSurreal`` import
in beir_runner.py transitively imports ``hydrag.surreal_adapter``,
which raises ImportError at import time whenever the ``surrealdb``
package isn't installed -- breaking every import of
``hydrag_benchmark.beir_runner`` (including this test suite's own
``tests/test_beir.py``) in the real CI environment (``pip install -e
".[dev]"``, no surrealdb extra).
"""

from __future__ import annotations


def test_beir_runner_importable_without_surrealdb() -> None:
    """Merely importing the module (as every test file that uses it
    does, including test_beir.py) must succeed regardless of whether
    surrealdb is installed."""
    import hydrag_benchmark.beir_runner as beir_runner
    assert hasattr(beir_runner, "run_beir_benchmark")


def test_head_d_surreal_still_importable_directly() -> None:
    """The module itself is untouched -- it's still importable on its
    own when surrealdb *is* available (this venv doesn't have it, so
    this asserts the expected failure mode is unchanged: ImportError,
    not some other breakage from the lazy-import refactor)."""
    import pytest

    with pytest.raises(ImportError):
        import hydrag_benchmark.heads.head_d_surreal  # noqa: F401
