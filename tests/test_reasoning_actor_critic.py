from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_graph import dedup, reasoning, storage


class ReasoningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = storage.connect(Path(self.tmp.name) / "rs.db")
        tid = storage.create_topic(self.conn, "x")
        self.oid = storage.create_objective(self.conn, tid, "what is fast")
        self.sid1, _ = dedup.upsert_source(
            self.conn, self.oid, None, None,
            "Llama 3 reaches 18 tokens per second on a 16GB Mac mini using metal.",
        )
        self.sid2, _ = dedup.upsert_source(
            self.conn, self.oid, None, None,
            "Mistral 7B reaches 22 tokens per second on the same hardware.",
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_actor_produces_thinking_referencing_sources(self) -> None:
        sources = [storage.get_source(self.conn, self.sid1), storage.get_source(self.conn, self.sid2)]
        text, cited = reasoning.actor_propose(sources, "what is fast")
        self.assertIn(str(self.sid1), text)
        self.assertEqual(set(cited), {self.sid1, self.sid2})

    def test_critic_accepts_clean_thinking(self) -> None:
        sources = [storage.get_source(self.conn, self.sid1)]
        text, cited = reasoning.actor_propose(sources, "q")
        verdict = reasoning.critic_verify(self.conn, text, cited)
        self.assertTrue(verdict.accepted, verdict.reasons)

    def test_critic_rejects_missing_source(self) -> None:
        verdict = reasoning.critic_verify(self.conn, "any text", [99999])
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("does not exist" in r for r in verdict.reasons))

    def test_critic_rejects_excessive_quote(self) -> None:
        big_quote = "Llama 3 reaches 18 tokens per second on a 16GB Mac mini using metal." * 10
        verdict = reasoning.critic_verify(self.conn, big_quote, [self.sid1])
        self.assertFalse(verdict.accepted)

    def test_critic_rejects_offset_misaligned_quote(self) -> None:
        # Build a long source whose verbatim slice starting at offset 13 (NOT 0)
        # exceeds the 200-char quote budget.
        long_body = (
            "PREFIXPREFIX_" +  # 13 chars of unrelated prefix at offset 0
            "Llama 3 reaches eighteen tokens per second on a 16GB Mac mini using metal "
            "acceleration; Mistral 7B reaches twenty-two tokens per second on the same "
            "hardware according to recent benchmarking notes from independent reviewers "
            "all over the internet today."
        )
        sid, _ = dedup.upsert_source(self.conn, self.oid, None, None, long_body)
        slice_start = 13
        slice_len = 210
        verbatim = long_body[slice_start : slice_start + slice_len]
        self.assertEqual(len(verbatim), slice_len)
        thinking = "Some preamble. " + verbatim + " Some trailing analysis."
        verdict = reasoning.critic_verify(self.conn, thinking, [sid])
        self.assertFalse(verdict.accepted)
        self.assertTrue(
            any("quoted verbatim" in r for r in verdict.reasons),
            verdict.reasons,
        )

    def test_critic_accepts_short_offset_quote(self) -> None:
        # A 38-char verbatim run at source offset 7 must NOT trip the quote rule
        # (under MIN_VERBATIM_RUN=40 threshold).
        body = "ABCDEFG_short verbatim run here that is fine_TRAILING JUNK CONTENT"
        sid, _ = dedup.upsert_source(self.conn, self.oid, None, None, body)
        verbatim = body[7 : 7 + 38]
        self.assertEqual(len(verbatim), 38)
        thinking = f"- source {sid} keywords: alpha, beta\n{verbatim}"
        verdict = reasoning.critic_verify(self.conn, thinking, [sid])
        self.assertTrue(verdict.accepted, verdict.reasons)

    def test_rejected_thinking_not_persisted(self) -> None:
        before = len(storage.list_thinkings(self.conn, self.oid))
        verdict = reasoning.critic_verify(self.conn, "x", [99999])
        self.assertFalse(verdict.accepted)
        after = len(storage.list_thinkings(self.conn, self.oid))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
