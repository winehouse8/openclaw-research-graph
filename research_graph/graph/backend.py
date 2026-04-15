"""Graph backend protocol.

Any backend that implements these methods can be swapped into the
research_graph store layer without requiring changes above it. The
in-memory backend is used for tests and the default CLI; the Neo4j
backend takes over when the user points the factory at a live bolt
endpoint.
"""
from __future__ import annotations

from typing import Any, Protocol


class GraphBackend(Protocol):
    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    def create_node(self, label: str, properties: dict[str, Any]) -> int: ...

    def get_node(self, node_id: int) -> dict[str, Any] | None: ...

    def find_node(
        self, label: str, where: dict[str, Any]
    ) -> dict[str, Any] | None: ...

    def list_nodes(
        self, label: str, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    def update_node(self, node_id: int, properties: dict[str, Any]) -> None: ...

    def delete_node(self, node_id: int, cascade: bool = False) -> None: ...

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    def create_edge(
        self,
        src_id: int,
        rel: str,
        dst_id: int,
        properties: dict[str, Any] | None = None,
    ) -> None: ...

    def has_edge(self, src_id: int, rel: str, dst_id: int) -> bool: ...

    def out_neighbors(
        self,
        node_id: int,
        rel: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def in_neighbors(
        self,
        node_id: int,
        rel: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]: ...

    # ------------------------------------------------------------------
    # Path / multi-hop
    # ------------------------------------------------------------------
    def shortest_path(
        self,
        src_id: int,
        dst_id: int,
        rels: list[str] | None = None,
    ) -> list[int] | None: ...

    def neighborhood(
        self,
        node_id: int,
        hops: int,
        rels: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    # ------------------------------------------------------------------
    # Raw escape hatch
    # ------------------------------------------------------------------
    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...
