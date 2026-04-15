#!/usr/bin/env python3
"""Live demo of the Neo4jBackend against a real database.

Skips cleanly when ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD``
are not set, so running it on a machine without credentials is a no-op.
When set, it runs a small smoke scenario (create a topic, objective,
source, thinking; cite; walk) against the live Neo4j instance using the
official ``neo4j`` driver.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and password):
        print("skipping Neo4j live demo - set NEO4J_* env vars to run")
        return 0

    from research_graph import store  # noqa: E402
    from research_graph.graph.neo4j_backend import Neo4jBackend  # noqa: E402

    database = os.environ.get("NEO4J_DATABASE") or None
    backend = Neo4jBackend(uri, user, password, database=database)
    try:
        tid = store.create_topic(backend, "neo4j-demo-topic")
        oid = store.create_objective(backend, tid, "neo4j backend smoke")
        s1 = store.insert_source(
            backend, oid, "https://ex/x", "demo",
            "demo source content for neo4j backend", "neo4jdemo-h1", "q",
        )
        s2 = store.insert_source(
            backend, oid, "https://ex/y", "demo2",
            "second demo source content for neo4j", "neo4jdemo-h2", "q",
        )
        t1 = store.insert_thinking(
            backend, oid, "initial synthesis", "neo4jdemo-t1", [s1, s2], "actor"
        )
        t2 = store.insert_thinking(
            backend, oid, "refined synthesis", "neo4jdemo-t2", [s2], "actor",
            supersedes_id=t1,
        )
        print("== Neo4j demo ==")
        print(f"  topic       = {tid}")
        print(f"  objective   = {oid}")
        print(f"  sources     = {[s1, s2]}")
        print(f"  thinkings   = {[t1, t2]}")
        chain = store.supersession_chain(backend, t2)
        print(f"  chain(t2)   = {[int(c['id']) for c in chain]}")
        walk = store.four_hop_evidence_walk(backend, t1)
        print(f"  4hop start  = {int(walk['start']['id'])}")
        print(f"  4hop shared = {[int(x['id']) for x in walk['shared_sources']]}")
        # Cleanup
        store.delete_topic(backend, tid)
        print("  cleaned up demo nodes")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
