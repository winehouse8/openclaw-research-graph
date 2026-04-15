"""Backend selection based on environment variables.

Selection rules:

- ``OPENCLAW_GRAPH_BACKEND=neo4j`` plus ``NEO4J_URI`` / ``NEO4J_USER`` /
  ``NEO4J_PASSWORD`` -> Neo4jBackend
- otherwise -> InMemoryGraphBackend (default for tests and offline use)

The factory never touches a live Neo4j server unless the caller has
explicitly opted in via environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .backend import GraphBackend
from .inmemory import InMemoryGraphBackend


def get_default_backend(path: str | Path | None = None) -> GraphBackend:
    backend_name = os.environ.get("OPENCLAW_GRAPH_BACKEND", "").strip().lower()
    if backend_name == "neo4j":
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        if not (uri and user and password):
            raise ValueError(
                "OPENCLAW_GRAPH_BACKEND=neo4j requires NEO4J_URI, NEO4J_USER, "
                "and NEO4J_PASSWORD environment variables"
            )
        from .neo4j_backend import Neo4jBackend  # local import; optional dep
        database = os.environ.get("NEO4J_DATABASE") or None
        return Neo4jBackend(uri, user, password, database=database)

    backend = InMemoryGraphBackend()
    if path is not None:
        p = Path(path)
        if p.exists():
            backend.load_from_path(p)
        else:
            backend.bind_path(p)
    return backend
