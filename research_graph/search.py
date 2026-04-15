from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass
class SearchHit:
    url: str
    title: str
    content: str


class ExternalSearch(Protocol):
    def search(self, query: str) -> list[SearchHit]: ...


class OfflineFixtureSearch:
    def __init__(self, fixture_path: Path | None = None) -> None:
        if fixture_path is None:
            fixture_path = Path(__file__).parent / "_fixtures" / "search.json"
        self.fixture_path = fixture_path
        with open(fixture_path, "r", encoding="utf-8") as f:
            self._data: dict[str, list[dict]] = json.load(f)

    def search(self, query: str) -> list[SearchHit]:
        q = query.lower()
        # pick the fixture key with the most token overlap with the query
        best_key = None
        best_overlap = 0
        for key in self._data:
            if key == "default":
                continue
            overlap = sum(1 for tok in key.split() if tok in q)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
        hits = self._data.get(best_key) if best_key else self._data.get("default", [])
        return [SearchHit(url=h["url"], title=h["title"], content=h["content"]) for h in hits]


_BUILTIN_BACKENDS: dict[str, Callable[[], ExternalSearch]] = {
    "offline": OfflineFixtureSearch,
}


def register_backend(name: str, factory: Callable[[], ExternalSearch]) -> None:
    """Explicit extension point for trusted callers to register a backend."""
    _BUILTIN_BACKENDS[name] = factory


def get_default_search() -> ExternalSearch:
    name = os.environ.get("OPENCLAW_RESEARCH_SEARCH", "offline").strip() or "offline"
    if name not in _BUILTIN_BACKENDS:
        raise ValueError(
            f"unknown search backend {name!r}; allowed: {sorted(_BUILTIN_BACKENDS)}"
        )
    return _BUILTIN_BACKENDS[name]()
