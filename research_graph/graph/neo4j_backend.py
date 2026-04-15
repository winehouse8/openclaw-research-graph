"""Neo4j 5.x backend for the GraphBackend protocol.

This wraps the official ``neo4j`` python driver. All methods translate
into Cypher statements that keep the same node/edge semantics as the
in-memory backend. Stable identifiers are stored on an internal ``id``
property on every node so the Protocol's int-based ids survive round
trips through the database; we do not rely on ``elementId()`` which
depends on server internals.

The backend installs the expected uniqueness constraints on first
connection. They are idempotent (``IF NOT EXISTS``) so repeated
instantiation against the same database is safe.

This module is only imported when the user opts in via
``OPENCLAW_GRAPH_BACKEND=neo4j``; nothing above it in the call stack
takes a hard dependency on the ``neo4j`` package.
"""
from __future__ import annotations

from typing import Any


class Neo4jBackend:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
    ) -> None:
        # Local import so importing this module file does not require the
        # driver to be installed unless the caller actually instantiates
        # the backend.
        from neo4j import GraphDatabase  # type: ignore

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._ensure_schema()
        self._seed_counter()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def _ensure_schema(self) -> None:
        constraints = [
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT objective_id IF NOT EXISTS FOR (n:Objective) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT thinking_id IF NOT EXISTS FOR (n:Thinking) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (n:Topic) REQUIRE n.name IS UNIQUE",
            # Counter singleton used to hand out stable numeric ids.
            "CREATE CONSTRAINT counter_name IF NOT EXISTS FOR (n:_Counter) REQUIRE n.name IS UNIQUE",
        ]
        with self._session() as s:
            for c in constraints:
                s.run(c)

    def _seed_counter(self) -> None:
        with self._session() as s:
            s.run(
                "MERGE (c:_Counter {name: 'node'}) "
                "ON CREATE SET c.value = 0"
            )

    def _next_id(self) -> int:
        with self._session() as s:
            rec = s.run(
                "MATCH (c:_Counter {name: 'node'}) "
                "SET c.value = c.value + 1 "
                "RETURN c.value AS v"
            ).single()
            return int(rec["v"])

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    def create_node(self, label: str, properties: dict[str, Any]) -> int:
        if label not in {"Topic", "Objective", "Source", "Thinking"}:
            raise ValueError(f"unsupported node label: {label}")
        nid = self._next_id()
        props = dict(properties)
        props["id"] = nid
        # Cypher node labels cannot be parameterised so we splice the
        # validated label directly into the query.
        cypher = f"CREATE (n:{label}) SET n += $props RETURN n.id AS id"
        with self._session() as s:
            s.run(cypher, props=props)
        return nid

    def get_node(self, node_id: int) -> dict[str, Any] | None:
        cypher = (
            "MATCH (n) WHERE n.id = $id AND NOT n:_Counter "
            "RETURN n, labels(n) AS labels"
        )
        with self._session() as s:
            rec = s.run(cypher, id=int(node_id)).single()
        return self._flatten_record(rec)

    def find_node(
        self, label: str, where: dict[str, Any]
    ) -> dict[str, Any] | None:
        clauses, params = self._where_clause(where)
        cypher = f"MATCH (n:{label}) {clauses} RETURN n, labels(n) AS labels LIMIT 1"
        with self._session() as s:
            rec = s.run(cypher, **params).single()
        return self._flatten_record(rec)

    def list_nodes(
        self, label: str, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        clauses, params = self._where_clause(where or {})
        cypher = (
            f"MATCH (n:{label}) {clauses} "
            "RETURN n, labels(n) AS labels ORDER BY n.id"
        )
        with self._session() as s:
            rows = list(s.run(cypher, **params))
        return [self._flatten_record(r) for r in rows if r is not None]

    def update_node(self, node_id: int, properties: dict[str, Any]) -> None:
        cypher = "MATCH (n) WHERE n.id = $id SET n += $props"
        with self._session() as s:
            s.run(cypher, id=int(node_id), props=dict(properties))

    def delete_node(self, node_id: int, cascade: bool = False) -> None:
        if cascade:
            # Cascade through ownership edges only.
            cypher = (
                "MATCH (n) WHERE n.id = $id "
                "OPTIONAL MATCH (n)-[:HAS_OBJECTIVE|HAS_SOURCE|HAS_THINKING*0..]->(m) "
                "DETACH DELETE n, m"
            )
        else:
            cypher = "MATCH (n) WHERE n.id = $id DETACH DELETE n"
        with self._session() as s:
            s.run(cypher, id=int(node_id))

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
        if rel not in {
            "HAS_OBJECTIVE",
            "HAS_SOURCE",
            "HAS_THINKING",
            "CITES",
            "SUPERSEDES",
            "REUSES",
        }:
            raise ValueError(f"unsupported relationship type: {rel}")
        cypher = (
            "MATCH (a), (b) WHERE a.id = $src AND b.id = $dst "
            f"MERGE (a)-[r:{rel}]->(b) SET r += $props"
        )
        with self._session() as s:
            s.run(
                cypher,
                src=int(src_id),
                dst=int(dst_id),
                props=dict(properties or {}),
            )

    def has_edge(self, src_id: int, rel: str, dst_id: int) -> bool:
        cypher = (
            "MATCH (a)-[r]->(b) WHERE a.id = $src AND b.id = $dst AND type(r) = $rel "
            "RETURN count(r) AS c"
        )
        with self._session() as s:
            rec = s.run(
                cypher, src=int(src_id), dst=int(dst_id), rel=rel
            ).single()
        return int(rec["c"]) > 0

    def out_neighbors(
        self,
        node_id: int,
        rel: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        rel_part = f":{rel}" if rel else ""
        label_part = f":{label}" if label else ""
        cypher = (
            f"MATCH (a)-[r{rel_part}]->(b{label_part}) WHERE a.id = $id "
            "RETURN b AS n, labels(b) AS labels ORDER BY b.id"
        )
        with self._session() as s:
            rows = list(s.run(cypher, id=int(node_id)))
        return [self._flatten_record(r) for r in rows]

    def in_neighbors(
        self,
        node_id: int,
        rel: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        rel_part = f":{rel}" if rel else ""
        label_part = f":{label}" if label else ""
        cypher = (
            f"MATCH (a{label_part})-[r{rel_part}]->(b) WHERE b.id = $id "
            "RETURN a AS n, labels(a) AS labels ORDER BY a.id"
        )
        with self._session() as s:
            rows = list(s.run(cypher, id=int(node_id)))
        return [self._flatten_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Path / multi-hop
    # ------------------------------------------------------------------
    def shortest_path(
        self,
        src_id: int,
        dst_id: int,
        rels: list[str] | None = None,
    ) -> list[int] | None:
        rel_filter = "|".join(rels) if rels else ""
        rel_part = f":{rel_filter}" if rel_filter else ""
        cypher = (
            f"MATCH (a), (b), p = shortestPath((a)-[{rel_part}*..15]->(b)) "
            "WHERE a.id = $src AND b.id = $dst "
            "RETURN [n IN nodes(p) | n.id] AS ids"
        )
        with self._session() as s:
            rec = s.run(cypher, src=int(src_id), dst=int(dst_id)).single()
        if rec is None:
            return None
        return [int(x) for x in rec["ids"]]

    def neighborhood(
        self,
        node_id: int,
        hops: int,
        rels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        rel_filter = "|".join(rels) if rels else ""
        rel_part = f":{rel_filter}" if rel_filter else ""
        cypher = (
            f"MATCH (a)-[{rel_part}*1..{int(hops)}]-(b) WHERE a.id = $id AND b.id <> $id "
            "RETURN DISTINCT b AS n, labels(b) AS labels ORDER BY b.id"
        )
        with self._session() as s:
            rows = list(s.run(cypher, id=int(node_id)))
        return [self._flatten_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Raw escape hatch
    # ------------------------------------------------------------------
    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = list(s.run(query, **(params or {})))
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flatten_record(self, rec) -> dict[str, Any] | None:
        if rec is None:
            return None
        node = rec["n"]
        labels = list(rec["labels"])
        label = next(
            (l for l in labels if l in {"Topic", "Objective", "Source", "Thinking"}),
            labels[0] if labels else "",
        )
        out: dict[str, Any] = dict(node)
        out["label"] = label
        if "id" in out:
            out["id"] = int(out["id"])
        return out

    def _where_clause(
        self, where: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        if not where:
            return "", {}
        parts: list[str] = []
        params: dict[str, Any] = {}
        for k, v in where.items():
            key = f"w_{k}"
            parts.append(f"n.{k} = ${key}")
            params[key] = v
        return "WHERE " + " AND ".join(parts), params
