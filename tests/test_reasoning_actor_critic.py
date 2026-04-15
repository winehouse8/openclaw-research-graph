from __future__ import annotations

import unittest

from research_graph import dedup, reasoning, store
from research_graph.graph import InMemoryGraphBackend


class ReasoningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "x")
        self.oid = store.create_objective(self.backend, tid, "what is fast")
        self.sid1, _ = dedup.upsert_source(
            self.backend, self.oid, None, None,
            "Llama 3 reaches 18 tokens per second on a 16GB Mac mini using metal.",
        )
        self.sid2, _ = dedup.upsert_source(
            self.backend, self.oid, None, None,
            "Mistral 7B reaches 22 tokens per second on the same hardware.",
        )

    def test_actor_produces_thinking_referencing_sources(self) -> None:
        sources = [
            store.get_source(self.backend, self.sid1),
            store.get_source(self.backend, self.sid2),
        ]
        text, cited = reasoning.actor_propose(sources, "what is fast")
        self.assertIn(str(self.sid1), text)
        self.assertEqual(set(cited), {self.sid1, self.sid2})

    def test_critic_accepts_clean_thinking(self) -> None:
        sources = [store.get_source(self.backend, self.sid1)]
        text, cited = reasoning.actor_propose(sources, "q")
        verdict = reasoning.critic_verify(self.backend, text, cited)
        self.assertTrue(verdict.accepted, verdict.reasons)

    def test_critic_rejects_missing_source(self) -> None:
        verdict = reasoning.critic_verify(self.backend, "any text", [99999])
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("does not exist" in r for r in verdict.reasons))

    def test_critic_rejects_excessive_quote(self) -> None:
        big_quote = "Llama 3 reaches 18 tokens per second on a 16GB Mac mini using metal." * 10
        verdict = reasoning.critic_verify(self.backend, big_quote, [self.sid1])
        self.assertFalse(verdict.accepted)

    def test_critic_rejects_offset_misaligned_quote(self) -> None:
        long_body = (
            "PREFIXPREFIX_" +
            "Llama 3 reaches eighteen tokens per second on a 16GB Mac mini using metal "
            "acceleration; Mistral 7B reaches twenty-two tokens per second on the same "
            "hardware according to recent benchmarking notes from independent reviewers "
            "all over the internet today."
        )
        sid, _ = dedup.upsert_source(self.backend, self.oid, None, None, long_body)
        verbatim = long_body[13:13 + 210]
        thinking = "Some preamble. " + verbatim + " Some trailing analysis."
        verdict = reasoning.critic_verify(self.backend, thinking, [sid])
        self.assertFalse(verdict.accepted)
        self.assertTrue(
            any("quoted verbatim" in r for r in verdict.reasons), verdict.reasons
        )

    def test_critic_accepts_short_offset_quote(self) -> None:
        body = "ABCDEFG_short verbatim run here that is fine_TRAILING JUNK CONTENT"
        sid, _ = dedup.upsert_source(self.backend, self.oid, None, None, body)
        verbatim = body[7:7 + 38]
        thinking = f"- source {sid} keywords: alpha, beta\n{verbatim}"
        verdict = reasoning.critic_verify(self.backend, thinking, [sid])
        self.assertTrue(verdict.accepted, verdict.reasons)

    def test_rejected_thinking_not_persisted(self) -> None:
        before = len(store.list_thinkings(self.backend, self.oid))
        verdict = reasoning.critic_verify(self.backend, "x", [99999])
        self.assertFalse(verdict.accepted)
        after = len(store.list_thinkings(self.backend, self.oid))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
