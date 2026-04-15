from __future__ import annotations

import unittest

from research_graph import dedup, retrieval, store
from research_graph.graph import InMemoryGraphBackend


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(self.backend, tid, "best local llm")
        self.oid_other = store.create_objective(self.backend, tid, "other")
        dedup.upsert_source(
            self.backend, self.oid, None, None,
            "Llama 3 runs at 18 tokens per second on Mac mini with metal acceleration",
        )
        dedup.upsert_source(
            self.backend, self.oid, None, None,
            "Mistral 7B reaches 22 tokens per second on Apple Silicon hardware",
        )
        dedup.upsert_source(
            self.backend, self.oid, None, None,
            "Cooking pasta requires boiling water and salt",
        )
        dedup.upsert_source(
            self.backend, self.oid_other, None, None,
            "Llama 3 also runs on Linux with CUDA acceleration",
        )
        dedup.upsert_thinking(
            self.backend, self.oid, "Local LLM speeds favor Mistral on Mac", [], "actor"
        )

    def test_source_ranking(self) -> None:
        rows = retrieval.search_sources(
            self.backend, "llama mac mini metal", objective_id=self.oid, top_k=3
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("Llama", rows[0]["content"])
        self.assertNotIn("pasta", rows[0]["content"])

    def test_filter_by_objective(self) -> None:
        rows = retrieval.search_sources(
            self.backend, "Llama", objective_id=self.oid
        )
        for r in rows:
            self.assertEqual(r["objective_id"], self.oid)

    def test_separate_thinking_retrieval(self) -> None:
        rows = retrieval.search_thinkings(
            self.backend, "mistral mac", objective_id=self.oid
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("Mistral", rows[0]["content"])
        src_rows = retrieval.search_sources(
            self.backend, "mistral mac", objective_id=self.oid
        )
        for r in src_rows:
            self.assertIn("content", r)
            self.assertIn("url", r)

    def test_relevance_threshold(self) -> None:
        rows = retrieval.search_sources(
            self.backend, "completely unrelated quantum biology",
            objective_id=self.oid, min_score=0.5,
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
