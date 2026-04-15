from __future__ import annotations

import unittest

from research_graph import store
from research_graph.graph import InMemoryGraphBackend
from research_graph.orchestrator import Orchestrator
from research_graph.search import OfflineFixtureSearch, SearchHit


class StubSearch:
    def __init__(self) -> None:
        self.batches: list[list[SearchHit]] = [
            [
                SearchHit("u1", "t1", "Llama 3 runs at 18 tokens per second on Mac mini metal"),
                SearchHit("u2", "t2", "Mistral 7B reaches 22 tokens per second on Apple Silicon"),
            ],
            [
                SearchHit("u3", "t3", "Qwen 2.5 7B Instruct outperforms older Llama models on Mac mini"),
            ],
        ]
        self.calls = 0

    def search(self, query: str) -> list[SearchHit]:
        idx = min(self.calls, len(self.batches) - 1)
        self.calls += 1
        return self.batches[idx]


class OrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(
            self.backend, tid, "best local llm on 16gb mac mini"
        )

    def test_cold_start_then_memory_augmented(self) -> None:
        """Spec contract (spec.md L57-59): cold-start is external-only;
        memory-augmented BLENDS stored memory with NEW external info on
        EVERY run. The previous version of this test asserted the
        opposite — that the second run (no flags) produced zero new
        sources — which ratified a bug where the daily cron use case
        was a no-op after day 1. We now assert the spec-compliant
        behavior: r2 picks up the new hit from batch[1]."""
        stub = StubSearch()
        orch = Orchestrator(self.backend, search=stub)

        r1 = orch.research(self.oid)
        self.assertEqual(r1.mode, "cold_start")
        self.assertEqual(len(r1.new_source_ids), 2)
        self.assertIsNotNone(r1.new_thinking_id)
        self.assertIsNone(r1.supersedes_id)

        # Default behaviour: external fetch happens on EVERY run, so
        # the new u3 hit from batch[1] must arrive automatically.
        r2 = orch.research(self.oid)
        self.assertEqual(r2.mode, "memory_augmented")
        self.assertEqual(len(r2.new_source_ids), 1)
        self.assertIsNotNone(r2.new_thinking_id)
        self.assertEqual(r2.supersedes_id, r1.new_thinking_id)

        th = store.get_thinking(self.backend, r2.new_thinking_id)
        self.assertEqual(th["supersedes_id"], r1.new_thinking_id)
        # Old thinking is preserved (not dropped) — supersedes is
        # provenance, not destructive.
        self.assertIsNotNone(store.get_thinking(self.backend, r1.new_thinking_id))

        # Opt-out path: a third run with `skip_external=True` MUST NOT
        # fetch external. This is the offline-replay / explicit-pause
        # contract.
        prev_calls = stub.calls
        r3 = orch.research(self.oid, skip_external=True)
        self.assertEqual(r3.mode, "memory_augmented")
        self.assertEqual(r3.new_source_ids, [])
        self.assertEqual(stub.calls, prev_calls,
                          "skip_external=True must not invoke the search backend")

    def test_offline_fixture_default(self) -> None:
        orch = Orchestrator(self.backend, search=OfflineFixtureSearch())
        r = orch.research(self.oid)
        self.assertEqual(r.mode, "cold_start")
        self.assertGreater(len(r.new_source_ids), 0)

    def test_memory_augmented_dedup_reports_reuse(self) -> None:
        """When a memory-augmented run brings in NO new evidence (the
        external backend keeps returning the same batch and dedup
        collapses everything), the orchestrator should report
        `reused_thinking_id` and write no REUSES edge for Branch B.

        We force the "no new evidence" condition via a stable stub that
        always returns the same hits. The previous version of this test
        passed by accident — it only worked because memory-augmented
        runs SILENTLY skipped external fetch entirely. Now we exercise
        the path correctly: dedup must reject every "new" hit on the
        second call because content_hash matches."""
        class StableStub:
            def __init__(self) -> None:
                self.calls = 0
                self._hits = [
                    SearchHit("u1", "t1", "Llama 3 runs at 18 tokens per second on Mac mini metal"),
                    SearchHit("u2", "t2", "Mistral 7B reaches 22 tokens per second on Apple Silicon"),
                ]
            def search(self, query: str) -> list[SearchHit]:
                self.calls += 1
                return list(self._hits)
        stub = StableStub()
        orch = Orchestrator(self.backend, search=stub)

        cold = orch.research(self.oid)
        self.assertEqual(cold.mode, "cold_start")
        cold_thinking_id = cold.new_thinking_id
        self.assertEqual(len(cold.new_source_ids), 2)

        again = orch.research(self.oid)
        self.assertEqual(again.mode, "memory_augmented")
        # Stable stub returns identical hits → dedup collapses every
        # one, so no new sources, no new thinking.
        self.assertEqual(again.new_source_ids, [])
        self.assertEqual(len(again.reused_source_ids), 2)
        self.assertIsNone(again.new_thinking_id)
        self.assertEqual(again.reused_thinking_id, cold_thinking_id)
        self.assertIsNone(again.supersedes_id)
        self.assertIsNotNone(store.get_thinking(self.backend, cold_thinking_id))
        # Branch B: dedup collapsed onto the latest-and-only thinking, so
        # NO REUSES edge is written (self-loops forbidden; there is no
        # older row to point at). ResearchResult still carries
        # reused_thinking_id so the caller can observe the no-op run.
        reuses = store.list_reuses(self.backend, self.oid)
        self.assertEqual(len(reuses), 0)


class SearchBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        import os
        self._prev = os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)

    def tearDown(self) -> None:
        import os
        os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)
        if self._prev is not None:
            os.environ["OPENCLAW_RESEARCH_SEARCH"] = self._prev

    def test_default_backend_is_offline(self) -> None:
        from research_graph.search import OfflineFixtureSearch, get_default_search
        backend = get_default_search()
        self.assertIsInstance(backend, OfflineFixtureSearch)

    def test_unknown_backend_rejected(self) -> None:
        import os
        from research_graph.search import get_default_search
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "nope"
        with self.assertRaises(ValueError):
            get_default_search()

    def test_registered_backend_is_returned(self) -> None:
        import os
        from research_graph.search import (
            OfflineFixtureSearch,
            get_default_search,
            register_backend,
        )

        class _Sentinel(OfflineFixtureSearch):
            pass

        register_backend("sentinel-test", _Sentinel)
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "sentinel-test"
        try:
            self.assertIsInstance(get_default_search(), _Sentinel)
        finally:
            from research_graph import search as _search_mod
            _search_mod._BUILTIN_BACKENDS.pop("sentinel-test", None)


if __name__ == "__main__":
    unittest.main()
