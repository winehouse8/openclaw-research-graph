"""Optional smoke test for the real Neo4jBackend.

This is skipped by default so ``python3 -m unittest discover`` runs on a
stock machine without touching anybody's Neo4j. It exercises the real
neo4j driver IFF the three ``OC_TEST_NEO4J_*`` environment variables are
set.
"""
from __future__ import annotations

import os
import unittest

REQ_ENV = ("OC_TEST_NEO4J_URI", "OC_TEST_NEO4J_USER", "OC_TEST_NEO4J_PASSWORD")
_missing = [k for k in REQ_ENV if not os.environ.get(k)]


@unittest.skipIf(
    _missing,
    f"Neo4j smoke skipped; set {', '.join(REQ_ENV)} to enable. Missing: {_missing}",
)
class Neo4jSmokeTest(unittest.TestCase):
    def test_create_edge_neighborhood_delete(self) -> None:
        from research_graph.graph.neo4j_backend import Neo4jBackend
        from research_graph.graph.model import (
            LABEL_SOURCE,
            LABEL_THINKING,
            LABEL_TOPIC,
            REL_CITES,
            REL_HAS_SOURCE,
        )

        uri = os.environ["OC_TEST_NEO4J_URI"]
        user = os.environ["OC_TEST_NEO4J_USER"]
        password = os.environ["OC_TEST_NEO4J_PASSWORD"]
        database = os.environ.get("OC_TEST_NEO4J_DATABASE")

        backend = Neo4jBackend(uri, user, password, database=database)
        try:
            t = backend.create_node(LABEL_TOPIC, {"name": "smoke-topic"})
            s = backend.create_node(LABEL_SOURCE, {
                "content": "body", "content_hash": "sm-hash", "objective_id": 0,
            })
            th = backend.create_node(LABEL_THINKING, {
                "content": "smoke", "content_hash": "sm-th", "objective_id": 0,
                "supports_source_ids": [s],
            })
            backend.create_edge(t, REL_HAS_SOURCE, s)
            backend.create_edge(th, REL_CITES, s)

            self.assertTrue(backend.has_edge(th, REL_CITES, s))
            nbh = backend.neighborhood(th, hops=2)
            nbh_ids = {int(n["id"]) for n in nbh}
            self.assertIn(s, nbh_ids)

            backend.delete_node(th)
            backend.delete_node(s)
            backend.delete_node(t)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
