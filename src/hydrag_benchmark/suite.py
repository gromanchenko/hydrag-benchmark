"""hydrag-benchmark suite model and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchCase:
    """A single benchmark query case."""

    id: str
    query: str
    relevant_phrases: list[str]
    tags: list[str] = field(default_factory=list)


@dataclass
class BenchSuite:
    """Parsed benchmark suite definition."""

    name: str
    version: str
    strategy: str
    n_results: int
    seed: int
    cases: list[BenchCase]
    description: str = ""

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        strategy_override: str | None = None,
        n_results_override: int | None = None,
        seed_override: int | None = None,
    ) -> BenchSuite:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = data.get("environment", {})
        cases = [
            BenchCase(
                id=c["id"],
                query=c["query"],
                relevant_phrases=c.get("relevant_phrases", []),
                tags=c.get("tags", []),
            )
            for c in data.get("cases", [])
        ]
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            strategy=strategy_override or env.get("strategy", "hydrag"),
            n_results=n_results_override or int(env.get("n_results", 5)),
            seed=seed_override or int(data.get("seed", 42)),
            cases=cases,
            description=data.get("description", ""),
        )
