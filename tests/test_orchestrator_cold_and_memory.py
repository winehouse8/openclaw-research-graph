from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_graph import storage
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
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = storage.connect(Path(self.tmp.name) / "o.db")
        tid = storage.create_topic(self.conn, "llm")
        self.oid = storage.create_objective(
            self.conn, tid, "best local llm on 16gb mac mini"
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_cold_start_then_memory_augmented(self) -> None:
        stub = StubSearch()
        orch = Orchestrator(self.conn, search=stub)

        r1 = orch.research(self.oid)
        self.assertEqual(r1.mode, "cold_start")
        self.assertEqual(len(r1.new_source_ids), 2)
        self.assertIsNotNone(r1.new_thinking_id)
        self.assertIsNone(r1.supersedes_id)

        # second run with no force_refresh: memory_augmented, no new sources, no new thinking
        r2 = orch.research(self.oid)
        self.assertEqual(r2.mode, "memory_augmented")
        self.assertEqual(r2.new_source_ids, [])

        # force refresh introduces new evidence -> supersession chain
        r3 = orch.research(self.oid, force_refresh=True)
        self.assertEqual(r3.mode, "memory_augmented")
        self.assertEqual(len(r3.new_source_ids), 1)
        self.assertIsNotNone(r3.new_thinking_id)
        self.assertEqual(r3.supersedes_id, r1.new_thinking_id)

        # supersession chain: thinking r3 points to r1
        th = storage.get_thinking(self.conn, r3.new_thinking_id)
        self.assertEqual(th["supersedes_id"], r1.new_thinking_id)
        # prior thinking still exists (history preserved)
        self.assertIsNotNone(storage.get_thinking(self.conn, r1.new_thinking_id))

    def test_offline_fixture_default(self) -> None:
        orch = Orchestrator(self.conn, search=OfflineFixtureSearch())
        r = orch.research(self.oid)
        self.assertEqual(r.mode, "cold_start")
        self.assertGreater(len(r.new_source_ids), 0)

    def test_memory_augmented_dedup_reports_reuse(self) -> None:
        stub = StubSearch()
        orch = Orchestrator(self.conn, search=stub)

        cold = orch.research(self.oid)
        self.assertEqual(cold.mode, "cold_start")
        self.assertIsNotNone(cold.new_thinking_id)
        cold_thinking_id = cold.new_thinking_id

        # Re-run with force_refresh=False so no new sources arrive; the actor
        # produces the same thinking which dedup collapses onto cold_thinking_id.
        again = orch.research(self.oid, force_refresh=False)
        self.assertEqual(again.mode, "memory_augmented")
        self.assertIsNone(again.new_thinking_id)
        self.assertEqual(again.reused_thinking_id, cold_thinking_id)
        # Prior thinking row still exists.
        self.assertIsNotNone(storage.get_thinking(self.conn, cold_thinking_id))


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
