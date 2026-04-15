from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_graph import dedup, retrieval, storage


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = storage.connect(Path(self.tmp.name) / "r.db")
        tid = storage.create_topic(self.conn, "llm")
        self.oid = storage.create_objective(self.conn, tid, "best local llm")
        self.oid_other = storage.create_objective(self.conn, tid, "other")
        dedup.upsert_source(
            self.conn, self.oid, None, None,
            "Llama 3 runs at 18 tokens per second on Mac mini with metal acceleration",
        )
        dedup.upsert_source(
            self.conn, self.oid, None, None,
            "Mistral 7B reaches 22 tokens per second on Apple Silicon hardware",
        )
        dedup.upsert_source(
            self.conn, self.oid, None, None,
            "Cooking pasta requires boiling water and salt",
        )
        dedup.upsert_source(
            self.conn, self.oid_other, None, None,
            "Llama 3 also runs on Linux with CUDA acceleration",
        )
        dedup.upsert_thinking(
            self.conn, self.oid, "Local LLM speeds favor Mistral on Mac", [], "actor"
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_source_ranking(self) -> None:
        rows = retrieval.search_sources(
            self.conn, "llama mac mini metal", objective_id=self.oid, top_k=3
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("Llama", rows[0]["content"])
        # cooking row should not outrank
        self.assertNotIn("pasta", rows[0]["content"])

    def test_filter_by_objective(self) -> None:
        rows = retrieval.search_sources(self.conn, "Llama", objective_id=self.oid)
        for r in rows:
            self.assertEqual(r["objective_id"], self.oid)

    def test_separate_thinking_retrieval(self) -> None:
        rows = retrieval.search_thinkings(self.conn, "mistral mac", objective_id=self.oid)
        self.assertEqual(len(rows), 1)
        self.assertIn("Mistral", rows[0]["content"])
        # source query must not return thinking and vice versa
        src_rows = retrieval.search_sources(self.conn, "mistral mac", objective_id=self.oid)
        for r in src_rows:
            self.assertIn("content", r)
            # ensure these came from sources table -- has a url column key
            self.assertIn("url", r)

    def test_relevance_threshold(self) -> None:
        rows = retrieval.search_sources(
            self.conn, "completely unrelated quantum biology",
            objective_id=self.oid, min_score=0.5,
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
