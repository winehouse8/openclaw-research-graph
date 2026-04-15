from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_graph import storage
from research_graph.dedup import content_hash


class StorageCRUQDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.conn = storage.connect(self.db)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_topic_crud(self) -> None:
        tid = storage.create_topic(self.conn, "llm")
        self.assertEqual(storage.get_topic(self.conn, tid)["name"], "llm")
        # idempotent on duplicate name
        self.assertEqual(storage.create_topic(self.conn, "llm"), tid)
        self.assertEqual(len(storage.list_topics(self.conn)), 1)

    def test_objective_crud_and_query(self) -> None:
        tid = storage.create_topic(self.conn, "llm")
        oid = storage.create_objective(self.conn, tid, "best local llm?")
        obj = storage.get_objective(self.conn, oid)
        self.assertEqual(obj["question"], "best local llm?")
        storage.update_objective(self.conn, oid, status="researched")
        self.assertEqual(storage.get_objective(self.conn, oid)["status"], "researched")
        rows = storage.query_objectives(self.conn, topic_id=tid, status="researched")
        self.assertEqual(len(rows), 1)
        rows = storage.query_objectives(self.conn, status="open")
        self.assertEqual(rows, [])

    def test_source_and_thinking_crud(self) -> None:
        tid = storage.create_topic(self.conn, "llm")
        oid = storage.create_objective(self.conn, tid, "q?")
        sid = storage.insert_source(
            self.conn, oid, "http://x", "t", "hello world content", content_hash("hello world content")
        )
        self.assertEqual(storage.get_source(self.conn, sid)["title"], "t")
        tid2 = storage.insert_thinking(
            self.conn, oid, "thought one", content_hash("thought one"), [sid], "actor"
        )
        th = storage.get_thinking(self.conn, tid2)
        self.assertEqual(th["supports_source_ids"], [sid])

    def test_cascade_delete(self) -> None:
        tid = storage.create_topic(self.conn, "llm")
        oid = storage.create_objective(self.conn, tid, "q?")
        sid = storage.insert_source(
            self.conn, oid, None, None, "abc", content_hash("abc")
        )
        storage.upsert_embedding(self.conn, "source", sid, {"abc": 1})
        thid = storage.insert_thinking(self.conn, oid, "x", content_hash("x"), [sid], "a")
        storage.upsert_embedding(self.conn, "thinking", thid, {"x": 1})
        storage.delete_objective(self.conn, oid)
        self.assertIsNone(storage.get_objective(self.conn, oid))
        self.assertIsNone(storage.get_source(self.conn, sid))
        self.assertIsNone(storage.get_thinking(self.conn, thid))
        self.assertEqual(storage.get_embeddings(self.conn, "source", [sid]), [])
        self.assertEqual(storage.get_embeddings(self.conn, "thinking", [thid]), [])

    def test_query_by_date_range(self) -> None:
        tid = storage.create_topic(self.conn, "x")
        oid = storage.create_objective(self.conn, tid, "q")
        rows = storage.query_objectives(self.conn, since="1970-01-01", until="2999-12-31")
        self.assertEqual(len(rows), 1)
        rows = storage.query_objectives(self.conn, since="2999-01-01")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
