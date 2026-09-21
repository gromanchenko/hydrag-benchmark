"""T-5061: coherent Python 3.12 floor across requires-python, classifiers,
mypy's python_version, CI's test matrix, and a version bump reflecting
the dropped runtime support (operator decision 2026-09-21, "consistent
switch to 3.12 everywhere within hydrags").
"""

from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return str(data["project"]["version"])


class TestPython312Floor:
    def test_requires_python_is_312_floor(self) -> None:
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        assert data["project"]["requires-python"] == ">=3.12"

    def test_no_310_or_311_classifiers(self) -> None:
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        classifiers = data["project"]["classifiers"]
        assert "Programming Language :: Python :: 3.10" not in classifiers
        assert "Programming Language :: Python :: 3.11" not in classifiers
        assert "Programming Language :: Python :: 3.12" in classifiers
        assert "Programming Language :: Python :: 3.13" in classifiers

    def test_mypy_python_version_is_312(self) -> None:
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        assert data["tool"]["mypy"]["python_version"] == "3.12"

    def test_ci_matrix_is_312_and_313_only(self) -> None:
        ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert '"3.10"' not in ci
        assert '"3.11"' not in ci
        assert '"3.12"' in ci
        assert '"3.13"' in ci

    def test_version_is_090(self) -> None:
        assert _pyproject_version() == "0.9.0"

    def test_version_py_agrees(self) -> None:
        from hydrag_benchmark import __version__
        assert __version__ == _pyproject_version()

    def test_changelog_states_310_311_dropped(self) -> None:
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
        assert "[0.9.0]" in changelog
        assert "3.10" in changelog and "3.11" in changelog
        lowered = changelog.lower()
        assert "no longer supported" in lowered or "dropped" in lowered


class TestStaticGatesExitZero:
    """T-5061: operator decision 2026-09-21 withdrew the earlier
    "no new errors" ratchet -- ruff check and mypy --strict must both
    exit 0, no baseline accepted. This test locks that in as a
    permanent regression guard (not just a one-time CI run)."""

    def test_ruff_check_exits_zero(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src/", "tests/"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_mypy_strict_exits_zero(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "src/hydrag_benchmark/"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
