from __future__ import annotations

import unittest

from research_graph import store
from research_graph.dedup import content_hash
from research_graph.graph import InMemoryGraphBackend


class StoreGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()

    def test_topic_idempotent_create(self) -> None:
        tid = store.create_topic(self.backend, "llm")
        self.assertEqual(store.create_topic(self.backend, "llm"), tid)
        self.assertEqual(len(store.list_topics(self.backend)), 1)

    def test_objective_crud_and_query(self) -> None:
        tid = store.create_topic(self.backend, "llm")
        oid = store.create_objective(self.backend, tid, "best local llm?")
        obj = store.get_objective(self.backend, oid)
        self.assertEqual(obj["question"], "best local llm?")
        store.update_objective(self.backend, oid, status="researched")
        self.assertEqual(store.get_objective(self.backend, oid)["status"], "researched")
        rows = store.query_objectives(self.backend, topic_id=tid, status="researched")
        self.assertEqual(len(rows), 1)
        self.assertEqual(store.query_objectives(self.backend, status="open"), [])

    def test_update_objective_rejects_unknown_field(self) -> None:
        tid = store.create_topic(self.backend, "llm")
        oid = store.create_objective(self.backend, tid, "q?")
        with self.assertRaises(ValueError):
            store.update_objective(self.backend, oid, status_evil="x")
        store.update_objective(self.backend, oid, status="researched")

    def test_source_and_thinking_crud(self) -> None:
        tid = store.create_topic(self.backend, "llm")
        oid = store.create_objective(self.backend, tid, "q?")
        sid = store.insert_source(
            self.backend, oid, "http://x", "t", "hello world", content_hash("hello world"), "q"
        )
        self.assertEqual(store.get_source(self.backend, sid)["title"], "t")
        thid = store.insert_thinking(
            self.backend, oid, "thought", content_hash("thought"), [sid], "actor"
        )
        th = store.get_thinking(self.backend, thid)
        self.assertEqual(th["supports_source_ids"], [sid])
        # CITES edge was created
        cited = self.backend.out_neighbors(thid, rel="CITES", label="Source")
        self.assertEqual([int(c["id"]) for c in cited], [sid])

    def test_cascade_delete(self) -> None:
        tid = store.create_topic(self.backend, "llm")
        oid = store.create_objective(self.backend, tid, "q?")
        sid = store.insert_source(
            self.backend, oid, None, None, "abc", content_hash("abc"), None
        )
        thid = store.insert_thinking(
            self.backend, oid, "x", content_hash("x"), [sid], "actor"
        )
        store.delete_topic(self.backend, tid)
        self.assertIsNone(store.get_objective(self.backend, oid))
        self.assertIsNone(store.get_source(self.backend, sid))
        self.assertIsNone(store.get_thinking(self.backend, thid))

    def test_supersede_and_reuse_edges(self) -> None:
        tid = store.create_topic(self.backend, "x")
        oid = store.create_objective(self.backend, tid, "q")
        sid = store.insert_source(
            self.backend, oid, None, None, "body", content_hash("body"), None
        )
        t1 = store.insert_thinking(
            self.backend, oid, "first", content_hash("first"), [sid], "actor"
        )
        t2 = store.insert_thinking(
            self.backend, oid, "second", content_hash("second"), [sid], "actor",
            supersedes_id=t1,
        )
        store.reuse(self.backend, t2, t1)
        chain = store.supersession_chain(self.backend, t2)
        self.assertEqual([int(c["id"]) for c in chain], [t2, t1])
        reuses = store.list_reuses(self.backend, oid)
        self.assertEqual(len(reuses), 1)
        self.assertEqual(int(reuses[0]["reuser"]["id"]), t2)
        self.assertEqual(int(reuses[0]["reused"]["id"]), t1)

    def test_date_range_filter(self) -> None:
        tid = store.create_topic(self.backend, "x")
        store.create_objective(self.backend, tid, "q")
        rows = store.query_objectives(self.backend, since="1970-01-01", until="2999-12-31")
        self.assertEqual(len(rows), 1)
        self.assertEqual(store.query_objectives(self.backend, since="2999-01-01"), [])


if __name__ == "__main__":
    unittest.main()
