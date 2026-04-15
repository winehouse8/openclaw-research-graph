from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_graph.graph import InMemoryGraphBackend
from research_graph.graph.model import (
    LABEL_OBJECTIVE,
    LABEL_SOURCE,
    LABEL_THINKING,
    LABEL_TOPIC,
    REL_CITES,
    REL_HAS_OBJECTIVE,
    REL_HAS_SOURCE,
    REL_HAS_THINKING,
    REL_SUPERSEDES,
)


class InMemoryBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.b = InMemoryGraphBackend()

    def test_create_and_get_node(self) -> None:
        nid = self.b.create_node(LABEL_TOPIC, {"name": "llm", "created_at": "2026"})
        row = self.b.get_node(nid)
        self.assertEqual(row["label"], LABEL_TOPIC)
        self.assertEqual(row["name"], "llm")
        self.assertEqual(row["id"], nid)

    def test_find_and_list_nodes(self) -> None:
        a = self.b.create_node(LABEL_TOPIC, {"name": "a"})
        b = self.b.create_node(LABEL_TOPIC, {"name": "b"})
        self.assertEqual(
            self.b.find_node(LABEL_TOPIC, {"name": "b"})["id"], b
        )
        rows = self.b.list_nodes(LABEL_TOPIC)
        self.assertEqual(sorted(int(r["id"]) for r in rows), sorted([a, b]))

    def test_update_and_delete(self) -> None:
        nid = self.b.create_node(LABEL_OBJECTIVE, {"question": "q", "status": "open"})
        self.b.update_node(nid, {"status": "researched"})
        self.assertEqual(self.b.get_node(nid)["status"], "researched")
        self.b.delete_node(nid)
        self.assertIsNone(self.b.get_node(nid))

    def test_edges_and_neighbors(self) -> None:
        t = self.b.create_node(LABEL_TOPIC, {"name": "t"})
        o = self.b.create_node(LABEL_OBJECTIVE, {"question": "q"})
        s = self.b.create_node(LABEL_SOURCE, {"content": "c", "content_hash": "h"})
        self.b.create_edge(t, REL_HAS_OBJECTIVE, o)
        self.b.create_edge(o, REL_HAS_SOURCE, s)
        self.assertTrue(self.b.has_edge(t, REL_HAS_OBJECTIVE, o))
        outs = self.b.out_neighbors(t, rel=REL_HAS_OBJECTIVE)
        self.assertEqual([int(r["id"]) for r in outs], [o])
        ins = self.b.in_neighbors(s, rel=REL_HAS_SOURCE)
        self.assertEqual([int(r["id"]) for r in ins], [o])

    def test_duplicate_edge_is_idempotent(self) -> None:
        a = self.b.create_node(LABEL_TOPIC, {"name": "a"})
        o = self.b.create_node(LABEL_OBJECTIVE, {"question": "q"})
        self.b.create_edge(a, REL_HAS_OBJECTIVE, o)
        self.b.create_edge(a, REL_HAS_OBJECTIVE, o)
        self.assertEqual(len(self.b.out_neighbors(a)), 1)

    def test_cascade_delete_via_store_rels(self) -> None:
        t = self.b.create_node(LABEL_TOPIC, {"name": "t"})
        o = self.b.create_node(LABEL_OBJECTIVE, {"question": "q"})
        s = self.b.create_node(LABEL_SOURCE, {"content": "c", "content_hash": "h"})
        th = self.b.create_node(LABEL_THINKING, {"content": "x", "content_hash": "y"})
        self.b.create_edge(t, REL_HAS_OBJECTIVE, o)
        self.b.create_edge(o, REL_HAS_SOURCE, s)
        self.b.create_edge(o, REL_HAS_THINKING, th)
        self.b.delete_node(t, cascade=True)
        self.assertIsNone(self.b.get_node(t))
        self.assertIsNone(self.b.get_node(o))
        self.assertIsNone(self.b.get_node(s))
        self.assertIsNone(self.b.get_node(th))

    def test_shortest_path(self) -> None:
        a = self.b.create_node(LABEL_TOPIC, {"name": "a"})
        o = self.b.create_node(LABEL_OBJECTIVE, {"question": "q"})
        s = self.b.create_node(LABEL_SOURCE, {"content": "c", "content_hash": "h"})
        self.b.create_edge(a, REL_HAS_OBJECTIVE, o)
        self.b.create_edge(o, REL_HAS_SOURCE, s)
        path = self.b.shortest_path(a, s)
        self.assertEqual(path, [a, o, s])
        self.assertIsNone(self.b.shortest_path(s, a))

    def test_neighborhood(self) -> None:
        a = self.b.create_node(LABEL_TOPIC, {"name": "a"})
        o = self.b.create_node(LABEL_OBJECTIVE, {"question": "q"})
        s = self.b.create_node(LABEL_SOURCE, {"content": "c", "content_hash": "h"})
        self.b.create_edge(a, REL_HAS_OBJECTIVE, o)
        self.b.create_edge(o, REL_HAS_SOURCE, s)
        ids = {int(r["id"]) for r in self.b.neighborhood(a, hops=2)}
        self.assertEqual(ids, {o, s})

    def test_execute_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.b.execute("MATCH (n) RETURN n")

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.json"
            t = self.b.create_node(LABEL_TOPIC, {"name": "llm"})
            o = self.b.create_node(LABEL_OBJECTIVE, {"question": "q"})
            self.b.create_edge(t, REL_HAS_OBJECTIVE, o)
            self.b.save_to_path(p)

            b2 = InMemoryGraphBackend()
            b2.load_from_path(p)
            self.assertEqual(b2.find_node(LABEL_TOPIC, {"name": "llm"})["id"], t)
            self.assertTrue(b2.has_edge(t, REL_HAS_OBJECTIVE, o))
            # next created node gets a fresh id that does not collide.
            nid = b2.create_node(LABEL_TOPIC, {"name": "fresh"})
            self.assertGreater(nid, o)

    def test_context_manager_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.json"
            b = InMemoryGraphBackend()
            b.bind_path(p)
            with b:
                b.create_node(LABEL_TOPIC, {"name": "x"})
            self.assertTrue(p.exists())
            b2 = InMemoryGraphBackend()
            b2.load_from_path(p)
            self.assertIsNotNone(b2.find_node(LABEL_TOPIC, {"name": "x"}))


if __name__ == "__main__":
    unittest.main()
