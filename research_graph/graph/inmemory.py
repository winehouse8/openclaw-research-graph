"""Pure-stdlib adjacency-list graph backend.

Used as the default for tests, the CLI, and the offline demo. JSON
persistence is provided so multiple CLI invocations sharing the same
``--db PATH`` argument preserve state across process boundaries.

Concurrency model (spec L79 long-running stability):
  - SINGLE WRITER, multiple READERS. The backend takes an exclusive
    file lock around `save_to_path` and a shared file lock around
    `load_from_path`. Two parallel writers will queue (the second
    waits on the first), preventing the JSON-corruption race that
    existed before this iteration. Reads can run concurrently with
    other reads.
  - Lock medium is `fcntl.flock` (Unix) on a sidecar `.lock` file
    next to the JSON. Windows is best-effort (no-op fallback).
  - The lock guards FILE I/O only, NOT in-memory state. Two backend
    instances in the same process still share their `_nodes` dict
    only via load/save — they are not thread-safe in-memory. The
    spec's use case is single-process per cron run, so this is
    sufficient.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

try:
    import fcntl  # type: ignore[import-not-found]
    _HAS_FLOCK = True
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    _HAS_FLOCK = False


class _FileLock:
    """Best-effort cross-process file lock (POSIX `fcntl.flock`).

    On systems without `fcntl` (Windows) this is a no-op so the
    backend stays usable in degraded mode. Use as a context manager:

        with _FileLock(path, exclusive=True):
            ...write JSON atomically...

    Lock is released on context exit. Lock file is `<path>.lock`.
    """

    def __init__(self, target: Path, *, exclusive: bool) -> None:
        self.target = Path(target)
        self.exclusive = exclusive
        self._fh = None  # type: ignore[var-annotated]

    def __enter__(self) -> "_FileLock":
        if not _HAS_FLOCK:
            return self
        lock_path = self.target.with_suffix(self.target.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Open for r+ if it exists, else w to create.
        self._fh = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115
        flag = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH  # type: ignore[union-attr]
        fcntl.flock(self._fh.fileno(), flag)  # type: ignore[union-attr]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        finally:
            try:
                self._fh.close()
            finally:
                self._fh = None

from .model import (
    LABEL_OBJECTIVE,
    LABEL_SOURCE,
    LABEL_THINKING,
    LABEL_TOPIC,
    REL_HAS_OBJECTIVE,
    REL_HAS_SOURCE,
    REL_HAS_THINKING,
)


class InMemoryGraphBackend:
    """Adjacency-list graph with optional JSON-on-disk persistence.

    The backend holds three structures:

    - ``nodes``: node_id -> {"label": str, "properties": dict}
    - ``out_edges``: node_id -> list[{"rel", "dst", "properties"}]
    - ``in_edges``: node_id -> list[{"rel", "src", "properties"}]

    Uniqueness constraints documented in the spec (Topic.name unique,
    Source.content_hash unique within an objective, Thinking.content_hash
    unique within an objective) are enforced in the store layer rather
    than here, because they depend on edge context not just node state.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, dict[str, Any]] = {}
        self._out: dict[int, list[dict[str, Any]]] = {}
        self._in: dict[int, list[dict[str, Any]]] = {}
        self._next_id: int = 1
        self._path: Path | None = None
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    def create_node(self, label: str, properties: dict[str, Any]) -> int:
        node_id = self._next_id
        self._next_id += 1
        props = dict(properties)
        props.setdefault("id", node_id)
        self._nodes[node_id] = {"label": label, "properties": props}
        self._out[node_id] = []
        self._in[node_id] = []
        self._dirty = True
        return node_id

    def get_node(self, node_id: int) -> dict[str, Any] | None:
        rec = self._nodes.get(int(node_id))
        if rec is None:
            return None
        return self._flatten(node_id, rec)

    def find_node(
        self, label: str, where: dict[str, Any]
    ) -> dict[str, Any] | None:
        for nid, rec in self._nodes.items():
            if rec["label"] != label:
                continue
            if all(rec["properties"].get(k) == v for k, v in where.items()):
                return self._flatten(nid, rec)
        return None

    def list_nodes(
        self, label: str, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for nid, rec in self._nodes.items():
            if rec["label"] != label:
                continue
            if where and not all(
                rec["properties"].get(k) == v for k, v in where.items()
            ):
                continue
            out.append(self._flatten(nid, rec))
        out.sort(key=lambda r: int(r["id"]))
        return out

    def update_node(self, node_id: int, properties: dict[str, Any]) -> None:
        rec = self._nodes.get(int(node_id))
        if rec is None:
            raise KeyError(f"unknown node: {node_id}")
        for k, v in properties.items():
            rec["properties"][k] = v
        self._dirty = True

    def delete_node(self, node_id: int, cascade: bool = False) -> None:
        node_id = int(node_id)
        if node_id not in self._nodes:
            return
        if cascade:
            # Cascade follows the outbound ownership edges so deleting a
            # Topic also removes its Objectives / Sources / Thinkings, but
            # does not follow CITES / SUPERSEDES / REUSES which are pure
            # references rather than ownership.
            owning_rels = {REL_HAS_OBJECTIVE, REL_HAS_SOURCE, REL_HAS_THINKING}
            victims: list[int] = []
            stack: list[int] = [node_id]
            seen: set[int] = set()
            while stack:
                nid = stack.pop()
                if nid in seen:
                    continue
                seen.add(nid)
                victims.append(nid)
                for e in list(self._out.get(nid, [])):
                    if e["rel"] in owning_rels:
                        stack.append(int(e["dst"]))
            for nid in victims:
                self._delete_single(nid)
        else:
            self._delete_single(node_id)
        self._dirty = True

    def _delete_single(self, node_id: int) -> None:
        # remove the node
        self._nodes.pop(node_id, None)
        # remove outbound edges and their inbound twins
        for e in self._out.pop(node_id, []):
            dst = int(e["dst"])
            self._in[dst] = [x for x in self._in.get(dst, []) if int(x["src"]) != node_id or x["rel"] != e["rel"]]
        # remove inbound edges and their outbound twins
        for e in self._in.pop(node_id, []):
            src = int(e["src"])
            self._out[src] = [x for x in self._out.get(src, []) if int(x["dst"]) != node_id or x["rel"] != e["rel"]]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    def create_edge(
        self,
        src_id: int,
        rel: str,
        dst_id: int,
        properties: dict[str, Any] | None = None,
    ) -> None:
        src_id = int(src_id)
        dst_id = int(dst_id)
        if src_id not in self._nodes:
            raise KeyError(f"unknown src node: {src_id}")
        if dst_id not in self._nodes:
            raise KeyError(f"unknown dst node: {dst_id}")
        if self.has_edge(src_id, rel, dst_id):
            return
        props = dict(properties or {})
        self._out.setdefault(src_id, []).append(
            {"rel": rel, "dst": dst_id, "properties": props}
        )
        self._in.setdefault(dst_id, []).append(
            {"rel": rel, "src": src_id, "properties": props}
        )
        self._dirty = True

    def has_edge(self, src_id: int, rel: str, dst_id: int) -> bool:
        for e in self._out.get(int(src_id), []):
            if e["rel"] == rel and int(e["dst"]) == int(dst_id):
                return True
        return False

    def out_neighbors(
        self,
        node_id: int,
        rel: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self._out.get(int(node_id), []):
            if rel is not None and e["rel"] != rel:
                continue
            nrec = self._nodes.get(int(e["dst"]))
            if nrec is None:
                continue
            if label is not None and nrec["label"] != label:
                continue
            out.append(self._flatten(int(e["dst"]), nrec))
        out.sort(key=lambda r: int(r["id"]))
        return out

    def in_neighbors(
        self,
        node_id: int,
        rel: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self._in.get(int(node_id), []):
            if rel is not None and e["rel"] != rel:
                continue
            nrec = self._nodes.get(int(e["src"]))
            if nrec is None:
                continue
            if label is not None and nrec["label"] != label:
                continue
            out.append(self._flatten(int(e["src"]), nrec))
        out.sort(key=lambda r: int(r["id"]))
        return out

    # ------------------------------------------------------------------
    # Path / multi-hop
    # ------------------------------------------------------------------
    def shortest_path(
        self,
        src_id: int,
        dst_id: int,
        rels: list[str] | None = None,
    ) -> list[int] | None:
        src_id = int(src_id)
        dst_id = int(dst_id)
        if src_id == dst_id:
            return [src_id]
        allowed = set(rels) if rels else None
        parents: dict[int, int | None] = {src_id: None}
        q: deque[int] = deque([src_id])
        while q:
            cur = q.popleft()
            for e in self._out.get(cur, []):
                if allowed is not None and e["rel"] not in allowed:
                    continue
                nxt = int(e["dst"])
                if nxt in parents:
                    continue
                parents[nxt] = cur
                if nxt == dst_id:
                    path: list[int] = []
                    node: int | None = nxt
                    while node is not None:
                        path.append(node)
                        node = parents[node]
                    return list(reversed(path))
                q.append(nxt)
        return None

    def neighborhood(
        self,
        node_id: int,
        hops: int,
        rels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        start = int(node_id)
        allowed = set(rels) if rels else None
        seen: dict[int, int] = {start: 0}
        q: deque[tuple[int, int]] = deque([(start, 0)])
        while q:
            cur, depth = q.popleft()
            if depth >= hops:
                continue
            for e in self._out.get(cur, []):
                if allowed is not None and e["rel"] not in allowed:
                    continue
                nxt = int(e["dst"])
                if nxt not in seen:
                    seen[nxt] = depth + 1
                    q.append((nxt, depth + 1))
            for e in self._in.get(cur, []):
                if allowed is not None and e["rel"] not in allowed:
                    continue
                nxt = int(e["src"])
                if nxt not in seen:
                    seen[nxt] = depth + 1
                    q.append((nxt, depth + 1))
        result: list[dict[str, Any]] = []
        for nid in sorted(seen):
            if nid == start:
                continue
            rec = self._nodes.get(nid)
            if rec is not None:
                result.append(self._flatten(nid, rec))
        return result

    # ------------------------------------------------------------------
    # Raw escape hatch
    # ------------------------------------------------------------------
    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "InMemoryGraphBackend does not parse Cypher; use the typed "
            "methods or go through research_graph.store which provides "
            "backend-agnostic multi-hop helpers."
        )

    def close(self) -> None:
        if self._path is not None and self._dirty:
            self.save_to_path(self._path)

    def __enter__(self) -> "InMemoryGraphBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_to_path(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "next_id": self._next_id,
            "nodes": [
                {
                    "id": nid,
                    "label": rec["label"],
                    "properties": rec["properties"],
                }
                for nid, rec in sorted(self._nodes.items())
            ],
            "edges": [
                {
                    "src": nid,
                    "rel": e["rel"],
                    "dst": int(e["dst"]),
                    "properties": e.get("properties", {}),
                }
                for nid in sorted(self._out)
                for e in self._out[nid]
            ],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Exclusive file lock around the tmp+rename atomic write so
        # parallel writers serialise instead of corrupting the JSON.
        # Also held during the rename so a reader can't see the new
        # path mid-replace. (The tmp file itself is only ours during
        # the lock window.)
        with _FileLock(path, exclusive=True):
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            tmp.replace(path)
        self._dirty = False

    def load_from_path(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            self._path = path
            return
        # Shared lock for reads — multiple loaders can run in parallel
        # but a writer (exclusive) blocks them until its rename lands.
        with _FileLock(path, exclusive=False):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        self._nodes = {}
        self._out = {}
        self._in = {}
        for n in payload.get("nodes", []):
            nid = int(n["id"])
            self._nodes[nid] = {
                "label": n["label"],
                "properties": dict(n.get("properties", {})),
            }
            self._out[nid] = []
            self._in[nid] = []
        for e in payload.get("edges", []):
            src = int(e["src"])
            dst = int(e["dst"])
            rel = e["rel"]
            props = dict(e.get("properties", {}))
            self._out.setdefault(src, []).append(
                {"rel": rel, "dst": dst, "properties": props}
            )
            self._in.setdefault(dst, []).append(
                {"rel": rel, "src": src, "properties": props}
            )
        self._next_id = int(payload.get("next_id", max(self._nodes) + 1 if self._nodes else 1))
        self._path = path
        self._dirty = False

    def bind_path(self, path: Path) -> None:
        """Attach a JSON path without loading. Used by factory for fresh dbs."""
        self._path = Path(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flatten(self, node_id: int, rec: dict[str, Any]) -> dict[str, Any]:
        out = dict(rec["properties"])
        out["id"] = int(node_id)
        out["label"] = rec["label"]
        return out

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._out.values())
