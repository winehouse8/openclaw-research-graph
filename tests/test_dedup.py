from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_graph import dedup, storage


class DedupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = storage.connect(Path(self.tmp.name) / "d.db")
        tid = storage.create_topic(self.conn, "t")
        self.oid = storage.create_objective(self.conn, tid, "q")

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_exact_hash_dedup_source(self) -> None:
        sid1, c1 = dedup.upsert_source(self.conn, self.oid, "u", "t", "hello world")
        sid2, c2 = dedup.upsert_source(self.conn, self.oid, "u", "t", "hello world")
        self.assertTrue(c1)
        self.assertFalse(c2)
        self.assertEqual(sid1, sid2)

    def test_near_dup_source(self) -> None:
        a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        b = "alpha beta gamma delta epsilon zeta eta theta iota lambda"  # 9/11 jaccard ~ 0.818
        c = "alpha beta gamma delta epsilon zeta eta theta iota kappa mu"  # 10/11 jaccard ~ 0.91
        sid_a, _ = dedup.upsert_source(self.conn, self.oid, None, None, a)
        sid_b, created_b = dedup.upsert_source(self.conn, self.oid, None, None, b)
        self.assertTrue(created_b, "0.818 should be below 0.85 threshold")
        sid_c, created_c = dedup.upsert_source(self.conn, self.oid, None, None, c)
        self.assertFalse(created_c, "0.91 should be above 0.85 threshold")
        self.assertEqual(sid_c, sid_a)

    def test_threshold_configurable(self) -> None:
        a = "one two three four five"
        b = "one two three four six"  # 4/6 = 0.667
        dedup.upsert_source(self.conn, self.oid, None, None, a)
        sid_b, created = dedup.upsert_source(
            self.conn, self.oid, None, None, b, threshold=0.5
        )
        self.assertFalse(created)

    def test_independent_dedup_spaces(self) -> None:
        text = "shared text body for both kinds"
        sid, _ = dedup.upsert_source(self.conn, self.oid, None, None, text)
        # thinking with same content must not collide with the source
        tid, created = dedup.upsert_thinking(
            self.conn, self.oid, text, [sid], author="actor"
        )
        self.assertTrue(created)
        # second insertion of same thinking deduped
        tid2, created2 = dedup.upsert_thinking(
            self.conn, self.oid, text, [sid], author="actor"
        )
        self.assertFalse(created2)
        self.assertEqual(tid, tid2)


if __name__ == "__main__":
    unittest.main()
