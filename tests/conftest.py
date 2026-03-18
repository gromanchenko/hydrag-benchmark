"""Shared test fixtures for hydrag-benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_suite_dir(tmp_path: Path) -> Path:
    """Return a temporary directory pre-seeded as a suite workspace."""
    (tmp_path / "suites").mkdir()
    (tmp_path / "results").mkdir()
    return tmp_path


@pytest.fixture()
def smoke_suite_yaml() -> str:
    """Minimal valid suite YAML for smoke tests."""
    return (
        "name: conftest-smoke\n"
        "version: '1.0'\n"
        "seed: 42\n"
        "description: Fixture suite.\n"
        "\n"
        "environment:\n"
        "  strategy: similarity\n"
        "  n_results: 3\n"
        "\n"
        "cases:\n"
        "  - id: fx-001\n"
        "    query: 'test query'\n"
        "    relevant_phrases:\n"
        "      - 'test'\n"
    )
