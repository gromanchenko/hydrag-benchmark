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

import subprocess
import sys
import textwrap


def _run_with_surrealdb_blocked(statement: str) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class _BlockSurrealDB(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "surrealdb" or fullname.startswith("surrealdb."):
                    raise ModuleNotFoundError("surrealdb blocked by regression test")
                return None

        sys.meta_path.insert(0, _BlockSurrealDB())
        {statement}
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_beir_runner_importable_without_surrealdb() -> None:
    """Merely importing the module (as every test file that uses it
    does, including test_beir.py) must succeed regardless of whether
    surrealdb is installed."""
    result = _run_with_surrealdb_blocked(
        "import hydrag_benchmark.beir_runner as module; "
        "assert hasattr(module, 'run_beir_benchmark')"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_head_d_surreal_still_importable_directly() -> None:
    """The module itself is untouched: requesting the SurrealDB-specific
    head while its optional dependency is unavailable still fails with
    the owning hydrag-core error instead of silently degrading."""
    result = _run_with_surrealdb_blocked(
        "import hydrag_benchmark.heads.head_d_surreal"
    )
    assert result.returncode != 0
    assert "requires the 'surrealdb' package" in result.stderr
