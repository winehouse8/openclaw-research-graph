from __future__ import annotations

import unittest

from research_graph import dedup, store
from research_graph.graph import InMemoryGraphBackend


class DedupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "t")
        self.oid = store.create_objective(self.backend, tid, "q")

    def test_exact_hash_dedup_source(self) -> None:
        sid1, c1 = dedup.upsert_source(self.backend, self.oid, "u", "t", "hello world")
        sid2, c2 = dedup.upsert_source(self.backend, self.oid, "u", "t", "hello world")
        self.assertTrue(c1)
        self.assertFalse(c2)
        self.assertEqual(sid1, sid2)

    def test_near_dup_source(self) -> None:
        a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        b = "alpha beta gamma delta epsilon zeta eta theta iota lambda"  # 9/11 jaccard ~ 0.818
        c = "alpha beta gamma delta epsilon zeta eta theta iota kappa mu"  # 10/11 ~ 0.91
        sid_a, _ = dedup.upsert_source(self.backend, self.oid, None, None, a)
        sid_b, created_b = dedup.upsert_source(self.backend, self.oid, None, None, b)
        self.assertTrue(created_b)
        sid_c, created_c = dedup.upsert_source(self.backend, self.oid, None, None, c)
        self.assertFalse(created_c)
        self.assertEqual(sid_c, sid_a)

    def test_threshold_configurable(self) -> None:
        a = "one two three four five"
        b = "one two three four six"
        dedup.upsert_source(self.backend, self.oid, None, None, a)
        sid_b, created = dedup.upsert_source(
            self.backend, self.oid, None, None, b, threshold=0.5
        )
        self.assertFalse(created)

    def test_independent_dedup_spaces(self) -> None:
        text = "shared text body for both kinds"
        sid, _ = dedup.upsert_source(self.backend, self.oid, None, None, text)
        tid, created = dedup.upsert_thinking(
            self.backend, self.oid, text, [sid], author="actor"
        )
        self.assertTrue(created)
        tid2, created2 = dedup.upsert_thinking(
            self.backend, self.oid, text, [sid], author="actor"
        )
        self.assertFalse(created2)
        self.assertEqual(tid, tid2)


if __name__ == "__main__":
    unittest.main()
