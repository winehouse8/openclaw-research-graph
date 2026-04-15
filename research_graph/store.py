"""Domain layer over the GraphBackend.

This module is the single entry point every other research_graph module
uses to read or mutate state. It never talks to SQLite; it speaks node /
edge primitives to whichever backend the factory handed back.

The multi-hop helpers at the bottom are written as pure adjacency-list
walks so they work identically against the in-memory backend and the
Neo4j backend. Each one has the equivalent Cypher in a comment so a
reader can see both sides of the interface without having to grep.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from .graph import (
    GraphBackend,
    LABEL_OBJECTIVE,
    LABEL_SOURCE,
    LABEL_THINKING,
    LABEL_TOPIC,
    REL_CITES,
    REL_HAS_OBJECTIVE,
    REL_HAS_SOURCE,
    REL_HAS_THINKING,
    REL_REUSES,
    REL_SUPERSEDES,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_OBJECTIVE_MUTABLE = {"question", "status"}


# ---------------------------------------------------------------------------
# Topic
# ---------------------------------------------------------------------------


def create_topic(backend: GraphBackend, name: str) -> int:
    existing = backend.find_node(LABEL_TOPIC, {"name": name})
    if existing is not None:
        return int(existing["id"])
    return backend.create_node(
        LABEL_TOPIC, {"name": name, "created_at": _now()}
    )


def get_topic(backend: GraphBackend, topic_id: int) -> dict | None:
    row = backend.get_node(int(topic_id))
    if row is None or row.get("label") != LABEL_TOPIC:
        return None
    return row


def get_topic_by_name(backend: GraphBackend, name: str) -> dict | None:
    return backend.find_node(LABEL_TOPIC, {"name": name})


def list_topics(backend: GraphBackend) -> list[dict]:
    return backend.list_nodes(LABEL_TOPIC)


def delete_topic(backend: GraphBackend, topic_id: int) -> None:
    # Walk Topic -> HAS_OBJECTIVE -> Objective -> HAS_SOURCE/HAS_THINKING
    # and delete every owned node. Cross-reference edges (CITES,
    # SUPERSEDES, REUSES) are removed automatically when the endpoints
    # vanish.
    objectives = backend.out_neighbors(int(topic_id), rel=REL_HAS_OBJECTIVE, label=LABEL_OBJECTIVE)
    for obj in objectives:
        delete_objective(backend, int(obj["id"]))
    backend.delete_node(int(topic_id), cascade=False)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def create_objective(
    backend: GraphBackend, topic_id: int, question: str
) -> int:
    now = _now()
    oid = backend.create_node(
        LABEL_OBJECTIVE,
        {
            "topic_id": int(topic_id),
            "question": question,
            "status": "open",
            "created_at": now,
            "updated_at": now,
        },
    )
    backend.create_edge(int(topic_id), REL_HAS_OBJECTIVE, oid)
    return oid


def get_objective(backend: GraphBackend, objective_id: int) -> dict | None:
    row = backend.get_node(int(objective_id))
    if row is None or row.get("label") != LABEL_OBJECTIVE:
        return None
    return row


def list_objectives(
    backend: GraphBackend, topic_id: int | None = None
) -> list[dict]:
    if topic_id is None:
        return backend.list_nodes(LABEL_OBJECTIVE)
    return backend.out_neighbors(int(topic_id), rel=REL_HAS_OBJECTIVE, label=LABEL_OBJECTIVE)


def query_objectives(
    backend: GraphBackend,
    topic_id: int | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    rows = list_objectives(backend, topic_id=topic_id)
    def keep(r: dict) -> bool:
        if status is not None and r.get("status") != status:
            return False
        if since is not None and r.get("created_at", "") < since:
            return False
        if until is not None and r.get("created_at", "") > until:
            return False
        return True
    return [r for r in rows if keep(r)]


def update_objective(
    backend: GraphBackend, objective_id: int, **fields: Any
) -> None:
    if not fields:
        return
    bad = set(fields) - _OBJECTIVE_MUTABLE
    if bad:
        raise ValueError(
            f"cannot update objective columns {sorted(bad)}; "
            f"allowed: {sorted(_OBJECTIVE_MUTABLE)}"
        )
    fields = dict(fields)
    fields["updated_at"] = _now()
    backend.update_node(int(objective_id), fields)


def delete_objective(backend: GraphBackend, objective_id: int) -> None:
    # Remove owned sources and thinkings first, then the objective.
    for s in backend.out_neighbors(int(objective_id), rel=REL_HAS_SOURCE, label=LABEL_SOURCE):
        backend.delete_node(int(s["id"]), cascade=False)
    for t in backend.out_neighbors(int(objective_id), rel=REL_HAS_THINKING, label=LABEL_THINKING):
        backend.delete_node(int(t["id"]), cascade=False)
    backend.delete_node(int(objective_id), cascade=False)


def objective_topic_id(backend: GraphBackend, objective_id: int) -> int | None:
    ins = backend.in_neighbors(int(objective_id), rel=REL_HAS_OBJECTIVE, label=LABEL_TOPIC)
    if not ins:
        return None
    return int(ins[0]["id"])


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def insert_source(
    backend: GraphBackend,
    objective_id: int,
    url: str | None,
    title: str | None,
    content: str,
    content_hash: str,
    search_query: str | None = None,
) -> int:
    sid = backend.create_node(
        LABEL_SOURCE,
        {
            "objective_id": int(objective_id),
            "url": url,
            "title": title,
            "content": content,
            "content_hash": content_hash,
            "fetched_at": _now(),
            "search_query": search_query,
        },
    )
    backend.create_edge(int(objective_id), REL_HAS_SOURCE, sid)
    return sid


def get_source(backend: GraphBackend, source_id: int) -> dict | None:
    row = backend.get_node(int(source_id))
    if row is None or row.get("label") != LABEL_SOURCE:
        return None
    return row


def list_sources(backend: GraphBackend, objective_id: int) -> list[dict]:
    return backend.out_neighbors(int(objective_id), rel=REL_HAS_SOURCE, label=LABEL_SOURCE)


def find_source_by_hash_in_objective(
    backend: GraphBackend, objective_id: int, content_hash: str
) -> dict | None:
    for row in list_sources(backend, objective_id):
        if row.get("content_hash") == content_hash:
            return row
    return None


def delete_source(backend: GraphBackend, source_id: int) -> None:
    backend.delete_node(int(source_id), cascade=False)


# ---------------------------------------------------------------------------
# Thinking
# ---------------------------------------------------------------------------


def insert_thinking(
    backend: GraphBackend,
    objective_id: int,
    content: str,
    content_hash: str,
    supports_source_ids: list[int],
    author: str,
    supersedes_id: int | None = None,
) -> int:
    tid = backend.create_node(
        LABEL_THINKING,
        {
            "objective_id": int(objective_id),
            "content": content,
            "content_hash": content_hash,
            # Keep the flat list on the node so legacy ResearchResult
            # consumers and downstream dict consumers still find it at
            # the same key, even though the canonical truth lives in
            # the CITES edges.
            "supports_source_ids": list(supports_source_ids),
            "author": author,
            "created_at": _now(),
            "supersedes_id": supersedes_id,
        },
    )
    backend.create_edge(int(objective_id), REL_HAS_THINKING, tid)
    for sid in supports_source_ids:
        backend.create_edge(tid, REL_CITES, int(sid))
    if supersedes_id is not None:
        backend.create_edge(tid, REL_SUPERSEDES, int(supersedes_id))
    return tid


def get_thinking(backend: GraphBackend, thinking_id: int) -> dict | None:
    row = backend.get_node(int(thinking_id))
    if row is None or row.get("label") != LABEL_THINKING:
        return None
    # Normalise supports_source_ids to list[int] regardless of backend.
    ssi = row.get("supports_source_ids") or []
    row["supports_source_ids"] = [int(x) for x in ssi]
    return row


def list_thinkings(backend: GraphBackend, objective_id: int) -> list[dict]:
    rows = backend.out_neighbors(int(objective_id), rel=REL_HAS_THINKING, label=LABEL_THINKING)
    for r in rows:
        ssi = r.get("supports_source_ids") or []
        r["supports_source_ids"] = [int(x) for x in ssi]
    return rows


def find_thinking_by_hash_in_objective(
    backend: GraphBackend, objective_id: int, content_hash: str
) -> dict | None:
    for row in list_thinkings(backend, objective_id):
        if row.get("content_hash") == content_hash:
            return row
    return None


def cite_source(
    backend: GraphBackend, thinking_id: int, source_id: int
) -> None:
    backend.create_edge(int(thinking_id), REL_CITES, int(source_id))


def supersede(backend: GraphBackend, new_id: int, old_id: int) -> None:
    backend.create_edge(int(new_id), REL_SUPERSEDES, int(old_id))


def reuse(backend: GraphBackend, reuser_id: int, reused_id: int) -> None:
    backend.create_edge(int(reuser_id), REL_REUSES, int(reused_id))


def get_thinking_with_citations(
    backend: GraphBackend, thinking_id: int
) -> dict:
    """Return the thinking node plus the cited Source nodes.

    Cypher equivalent:
        MATCH (t:Thinking {id: $tid})
        OPTIONAL MATCH (t)-[:CITES]->(s:Source)
        RETURN t, collect(s) AS sources
    """
    t = get_thinking(backend, thinking_id)
    if t is None:
        raise KeyError(f"unknown thinking: {thinking_id}")
    sources = backend.out_neighbors(int(thinking_id), rel=REL_CITES, label=LABEL_SOURCE)
    t["cited_sources"] = sources
    return t


def supersession_chain(
    backend: GraphBackend, thinking_id: int
) -> list[dict]:
    """Follow :SUPERSEDES pointers from ``thinking_id`` toward the root.

    Cypher equivalent:
        MATCH path = (t:Thinking {id: $tid})-[:SUPERSEDES*0..]->(anc:Thinking)
        RETURN [n IN nodes(path) | n] AS chain
    """
    chain: list[dict] = []
    seen: set[int] = set()
    cur: int | None = int(thinking_id)
    while cur is not None and cur not in seen:
        seen.add(cur)
        node = get_thinking(backend, cur)
        if node is None:
            break
        chain.append(node)
        parents = backend.out_neighbors(cur, rel=REL_SUPERSEDES, label=LABEL_THINKING)
        cur = int(parents[0]["id"]) if parents else None
    return chain


def list_reuses(
    backend: GraphBackend, objective_id: int | None = None
) -> list[dict]:
    """Return (reuser_thinking, reused_thinking) pairs via the REUSES edges.

    Cypher equivalent (optionally scoped by objective):
        MATCH (a:Thinking)-[:REUSES]->(b:Thinking)
        WHERE $oid IS NULL OR (a.objective_id = $oid AND b.objective_id = $oid)
        RETURN a, b
    """
    if objective_id is None:
        thinkings = backend.list_nodes(LABEL_THINKING)
    else:
        thinkings = list_thinkings(backend, int(objective_id))
    out: list[dict] = []
    for th in thinkings:
        for target in backend.out_neighbors(
            int(th["id"]), rel=REL_REUSES, label=LABEL_THINKING
        ):
            out.append({"reuser": th, "reused": target})
    return out


# ---------------------------------------------------------------------------
# Multi-hop graph helpers
# ---------------------------------------------------------------------------


def thinking_neighborhood(
    backend: GraphBackend, thinking_id: int
) -> dict:
    """One-hop view: cited sources, supersession parent, reuse parent.

    Cypher equivalent:
        MATCH (t:Thinking {id: $tid})
        OPTIONAL MATCH (t)-[:CITES]->(s:Source)
        OPTIONAL MATCH (t)-[:SUPERSEDES]->(prev:Thinking)
        OPTIONAL MATCH (t)-[:REUSES]->(reu:Thinking)
        RETURN t, collect(s) AS sources, prev, reu
    """
    node = get_thinking(backend, thinking_id)
    if node is None:
        raise KeyError(f"unknown thinking: {thinking_id}")
    sources = backend.out_neighbors(int(thinking_id), rel=REL_CITES, label=LABEL_SOURCE)
    prev = backend.out_neighbors(int(thinking_id), rel=REL_SUPERSEDES, label=LABEL_THINKING)
    reu = backend.out_neighbors(int(thinking_id), rel=REL_REUSES, label=LABEL_THINKING)
    return {
        "thinking": node,
        "cited_sources": sources,
        "supersedes": prev[0] if prev else None,
        "reuses": reu[0] if reu else None,
    }


def cocited_thinkings(
    backend: GraphBackend, thinking_id: int
) -> list[dict]:
    """Other thinkings sharing at least one cited Source with ``thinking_id``.

    Cypher equivalent:
        MATCH (t:Thinking {id: $tid})-[:CITES]->(s:Source)<-[:CITES]-(other:Thinking)
        WHERE other.id <> $tid
        RETURN DISTINCT other
    """
    cited = backend.out_neighbors(int(thinking_id), rel=REL_CITES, label=LABEL_SOURCE)
    seen: set[int] = set()
    out: list[dict] = []
    for s in cited:
        for other in backend.in_neighbors(
            int(s["id"]), rel=REL_CITES, label=LABEL_THINKING
        ):
            oid = int(other["id"])
            if oid == int(thinking_id) or oid in seen:
                continue
            seen.add(oid)
            out.append(other)
    out.sort(key=lambda r: int(r["id"]))
    return out


def four_hop_evidence_walk(
    backend: GraphBackend, thinking_id: int
) -> dict:
    """The 4-hop query from the analysis doc.

    Starting from a Thinking T0, walk CITES -> Source -> CITES^-1 -> Thinking T1
    -> CITES -> Source and return:
        - start: T0
        - shared_sources: sources cited by T0
        - related_thinkings: other thinkings that cite any of those sources
        - new_sources: sources cited by the related thinkings that T0 does
          NOT cite

    Cypher equivalent:
        MATCH (t0:Thinking {id: $tid})-[:CITES]->(s0:Source)<-[:CITES]-(t1:Thinking)
        WHERE t1.id <> t0.id
        OPTIONAL MATCH (t1)-[:CITES]->(s1:Source)
        WHERE NOT (t0)-[:CITES]->(s1)
        RETURN t0, collect(DISTINCT s0) AS shared,
               collect(DISTINCT t1) AS related,
               collect(DISTINCT s1) AS new_sources
    """
    start = get_thinking(backend, thinking_id)
    if start is None:
        raise KeyError(f"unknown thinking: {thinking_id}")

    own_sources = backend.out_neighbors(int(thinking_id), rel=REL_CITES, label=LABEL_SOURCE)
    own_source_ids = {int(s["id"]) for s in own_sources}

    related_map: dict[int, dict] = {}
    for s in own_sources:
        for other in backend.in_neighbors(int(s["id"]), rel=REL_CITES, label=LABEL_THINKING):
            oid = int(other["id"])
            if oid == int(thinking_id):
                continue
            related_map[oid] = other

    new_source_map: dict[int, dict] = {}
    for other_id in related_map:
        for s in backend.out_neighbors(other_id, rel=REL_CITES, label=LABEL_SOURCE):
            sid = int(s["id"])
            if sid in own_source_ids:
                continue
            new_source_map[sid] = s

    return {
        "start": start,
        "shared_sources": sorted(own_sources, key=lambda r: int(r["id"])),
        "related_thinkings": [
            related_map[k] for k in sorted(related_map)
        ],
        "new_sources": [new_source_map[k] for k in sorted(new_source_map)],
    }


def cross_topic_shared_sources(backend: GraphBackend) -> list[dict]:
    """Sources reachable from two or more distinct topics.

    Cypher equivalent:
        MATCH (t1:Topic)-[:HAS_OBJECTIVE]->(:Objective)-[:HAS_SOURCE]->(s:Source)
        MATCH (t2:Topic)-[:HAS_OBJECTIVE]->(:Objective)-[:HAS_SOURCE]->(s)
        WHERE t1.id < t2.id AND s.content_hash IS NOT NULL
        WITH s.content_hash AS h, collect(DISTINCT t1.id) + collect(DISTINCT t2.id) AS topics
        RETURN h, topics
    """
    topics = backend.list_nodes(LABEL_TOPIC)
    # For each topic, collect the set of content_hashes reachable through
    # its objectives' sources. Group by hash and keep hashes that show
    # up in more than one topic.
    hash_to_topics: dict[str, set[int]] = {}
    hash_to_sources: dict[str, list[dict]] = {}
    for topic in topics:
        topic_id = int(topic["id"])
        for obj in backend.out_neighbors(topic_id, rel=REL_HAS_OBJECTIVE, label=LABEL_OBJECTIVE):
            for src in backend.out_neighbors(int(obj["id"]), rel=REL_HAS_SOURCE, label=LABEL_SOURCE):
                h = src.get("content_hash")
                if not h:
                    continue
                hash_to_topics.setdefault(h, set()).add(topic_id)
                hash_to_sources.setdefault(h, []).append(src)
    out: list[dict] = []
    for h, tset in hash_to_topics.items():
        if len(tset) < 2:
            continue
        out.append(
            {
                "content_hash": h,
                "topic_ids": sorted(tset),
                "sources": hash_to_sources[h],
            }
        )
    out.sort(key=lambda r: r["content_hash"])
    return out


# ---------------------------------------------------------------------------
# Embeddings (term frequency bags) — stored as node properties to keep
# the graph self-contained. The retrieval layer reads / writes via these
# helpers instead of maintaining a separate embedding table.
# ---------------------------------------------------------------------------


def upsert_embedding(
    backend: GraphBackend, kind: str, object_id: int, terms: dict
) -> None:
    backend.update_node(int(object_id), {"_terms": dict(terms)})


def get_embeddings(
    backend: GraphBackend, kind: str, ids: Iterable[int] | None = None
) -> list[dict]:
    label = {"source": LABEL_SOURCE, "thinking": LABEL_THINKING}[kind]
    if ids is None:
        rows = backend.list_nodes(label)
    else:
        want = {int(x) for x in ids}
        rows = [r for r in backend.list_nodes(label) if int(r["id"]) in want]
    out: list[dict] = []
    for r in rows:
        terms = r.get("_terms") or {}
        out.append({"object_id": int(r["id"]), "terms": dict(terms)})
    return out
