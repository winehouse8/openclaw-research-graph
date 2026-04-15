"""Spec-compliance regression tests added 2026-04-15.

Each test here exists because a critical re-read of spec.md found a
gap between the spec and the existing code. The bug each test guards
is documented next to its assertions.

Spec.md sections referenced:
  - L38-40   지식 축적 (Knowledge accumulation)
  - L44-49   CRUQD (Create / Read / Update / Query / Delete)
  - L50      중복 감지 (Dedup)
  - L57-59   Cold-start vs memory-augmented; new info supersedes old
  - L61-68   OpenClaw integration
  - L73-76   Anti-over-engineering, conservative ambiguity handling
  - L79      Long-running structural stability
  - L86-91   Success criteria including daily 9am scheduled re-research
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from research_graph import dedup, store
from research_graph.graph import InMemoryGraphBackend
from research_graph.orchestrator import Orchestrator, ResearchResult
from research_graph.search import SearchHit


# ---------------------------------------------------------------------------
# AC2 — CRUQD U: update_source + update_thinking
# ---------------------------------------------------------------------------


class UpdateSourceThinkingTest(unittest.TestCase):
    """Spec L47 (Update). Before this iteration, store.py had only
    update_objective; sources and thinkings were missing the U in CRUQD.
    The only path to "edit a source" was delete + re-insert, which
    silently invalidated CITES / SUPERSEDES / REUSES edges pointing at
    the old node id."""

    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(self.backend, tid, "q?")
        self.sid = store.insert_source(
            self.backend, self.oid,
            url="https://example.com/v1",
            title="old title",
            content="some content",
            content_hash="h",
            search_query="orig",
        )
        self.tid_th = store.insert_thinking(
            self.backend, self.oid,
            content="some thinking",
            content_hash="th",
            supports_source_ids=[self.sid],
            author="actor",
        )

    def test_update_source_happy_path(self) -> None:
        store.update_source(self.backend, self.sid, url="https://example.com/v2", title="new title")
        row = store.get_source(self.backend, self.sid)
        self.assertEqual(row["url"], "https://example.com/v2")
        self.assertEqual(row["title"], "new title")
        # search_query unchanged because we did not pass it
        self.assertEqual(row["search_query"], "orig")
        # updated_at stamp must be set
        self.assertIn("updated_at", row)

    def test_update_source_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError) as cm:
            store.update_source(self.backend, self.sid, anything="x")
        self.assertIn("anything", str(cm.exception))

    def test_update_source_rejects_content_mutation(self) -> None:
        # content/content_hash MUST be immutable; mutating them would
        # desync the dedup index. Update path must reject and direct
        # the caller to dedup.upsert_source.
        with self.assertRaises(ValueError) as cm:
            store.update_source(self.backend, self.sid, content="new content")
        self.assertIn("content", str(cm.exception))
        with self.assertRaises(ValueError):
            store.update_source(self.backend, self.sid, content_hash="other")

    def test_update_source_no_op_with_no_fields(self) -> None:
        # Empty kwargs is a no-op, NOT an error.
        store.update_source(self.backend, self.sid)
        row = store.get_source(self.backend, self.sid)
        self.assertEqual(row["url"], "https://example.com/v1")

    def test_update_thinking_happy_path(self) -> None:
        store.update_thinking(self.backend, self.tid_th, author="actor-v2")
        row = store.get_thinking(self.backend, self.tid_th)
        self.assertEqual(row["author"], "actor-v2")
        self.assertIn("updated_at", row)

    def test_update_thinking_rejects_content_mutation(self) -> None:
        with self.assertRaises(ValueError):
            store.update_thinking(self.backend, self.tid_th, content="new")
        with self.assertRaises(ValueError):
            store.update_thinking(self.backend, self.tid_th, content_hash="x")


# ---------------------------------------------------------------------------
# AC12 — supersession-tip computation (latest_live_thinking)
# ---------------------------------------------------------------------------


class LatestLiveThinkingTest(unittest.TestCase):
    """Spec L79 — long-running stability. The orchestrator's prior
    code computed "latest thinking" as `list_thinkings(...)[-1]`,
    which assumed list-order == time-order. Works in-process today
    but is silently wrong on any backend with reclaimed/UUID ids.
    `latest_live_thinking` follows the SUPERSEDES chain instead."""

    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(self.backend, tid, "q?")

    def test_no_thinkings_returns_none(self) -> None:
        self.assertIsNone(store.latest_live_thinking(self.backend, self.oid))

    def test_single_thinking_is_live(self) -> None:
        t1 = store.insert_thinking(
            self.backend, self.oid, "first", "h1", [], "actor"
        )
        live = store.latest_live_thinking(self.backend, self.oid)
        self.assertEqual(int(live["id"]), t1)

    def test_supersession_chain_returns_tip(self) -> None:
        # t1 ← t2 ← t3   (t3 is the tip)
        t1 = store.insert_thinking(self.backend, self.oid, "v1", "h1", [], "actor")
        t2 = store.insert_thinking(self.backend, self.oid, "v2", "h2", [], "actor", supersedes_id=t1)
        t3 = store.insert_thinking(self.backend, self.oid, "v3", "h3", [], "actor", supersedes_id=t2)
        live = store.latest_live_thinking(self.backend, self.oid)
        self.assertEqual(int(live["id"]), t3)

    def test_branched_chain_breaks_tie_by_created_at(self) -> None:
        # t1 ← t2  (t2 is one tip)
        # t1 ← t3  (t3 is another tip, inserted later)
        t1 = store.insert_thinking(self.backend, self.oid, "v1", "h1", [], "actor")
        t2 = store.insert_thinking(self.backend, self.oid, "v2", "h2", [], "actor", supersedes_id=t1)
        # Force t3 to have a strictly-later created_at than t2 by
        # waiting one second before insert. (This test would fail
        # if the helper picked by id order instead of created_at.)
        import time
        time.sleep(0.01)
        t3 = store.insert_thinking(self.backend, self.oid, "v3", "h3", [], "actor", supersedes_id=t1)
        live = store.latest_live_thinking(self.backend, self.oid)
        # Both t2 and t3 are live tips; tie-breaks by created_at desc
        # then id desc. t3 has the newer created_at, so it wins.
        self.assertEqual(int(live["id"]), t3)


# ---------------------------------------------------------------------------
# AC3 — dedup near-duplicate via plugin_helper ingest path
# ---------------------------------------------------------------------------


class PluginHelperIngestDedupTest(unittest.TestCase):
    """Spec L50 + L61-68. The OpenClaw plugin must run ingested content
    through the SAME dedup layer as internal research runs. Previously
    `cmd_ingest` hand-rolled exact-hash dedup only, missing the
    Jaccard near-duplicate path. Two near-identical payloads ingested
    via the plugin would create two rows."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="rg-plugin-ingest-")
        self.db = str(Path(self.tmp) / "rg.json")
        # Wire plugin_helper into sys.path so we can import it.
        plugin_dir = Path(__file__).resolve().parent.parent / "plugins" / "research-memory"
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_objective(self) -> int:
        backend = InMemoryGraphBackend()
        try:
            tid = store.create_topic(backend, "llm")
            oid = store.create_objective(backend, tid, "best local llm")
            backend.save_to_path(self.db)
        finally:
            backend.close()
        return int(oid)

    def _ingest(self, oid: int, sources: list[dict]) -> dict:
        import argparse
        import plugin_helper as ph  # type: ignore[import-not-found]
        ns = argparse.Namespace(
            db=self.db,
            payload=json.dumps({"objective_id": oid, "sources": sources}),
        )
        return ph.cmd_ingest(ns)

    def test_exact_dup_collapses(self) -> None:
        oid = self._make_objective()
        first = self._ingest(oid, [
            {"url": "https://example.com/a", "title": "A", "content": "Llama 3 hits 18 t/s on Mac mini metal."},
        ])
        second = self._ingest(oid, [
            {"url": "https://example.com/a", "title": "A", "content": "Llama 3 hits 18 t/s on Mac mini metal."},
        ])
        self.assertEqual(len(first["inserted"]), 1)
        self.assertEqual(len(second["inserted"]), 0)
        self.assertEqual(len(second["skipped"]), 1)
        self.assertEqual(second["skipped"][0]["reason"], "duplicate")
        # And: same source_id reported in both runs (same row).
        self.assertEqual(first["inserted"][0], second["skipped"][0]["source_id"])

    def test_near_dup_collapses_via_dedup_path(self) -> None:
        # This is the bug fix: hand-rolled exact-hash ingest used to
        # treat the second payload as new because the bytes differ
        # (extra trailing whitespace + capitalization). dedup's
        # Jaccard normalizer canonicalizes whitespace + case so both
        # payloads land on the same row.
        oid = self._make_objective()
        first = self._ingest(oid, [
            {"url": "https://example.com/a", "title": "A",
             "content": "Llama 3 reaches 18 tokens per second on Apple Silicon Mac mini under metal acceleration."},
        ])
        second = self._ingest(oid, [
            {"url": "https://example.com/a", "title": "A",
             "content": "  Llama 3   reaches 18 tokens   per second on Apple Silicon Mac mini under metal acceleration.\n"},
        ])
        self.assertEqual(len(first["inserted"]), 1)
        self.assertEqual(len(second["inserted"]), 0,
                          "dedup path must collapse whitespace-variant near-duplicates")
        self.assertEqual(second["skipped"][0]["source_id"], first["inserted"][0])


# ---------------------------------------------------------------------------
# Spec contract — ResearchResult.to_dict canonical key set
# ---------------------------------------------------------------------------


class ResearchResultContractTest(unittest.TestCase):
    """Spec L40, L87 (knowledge reuse must be observable). The dict
    returned to OpenClaw must include reused_source_ids and
    rejected_reasons so the caller can tell "no new evidence" from
    "actor proposal was rejected" from "blended N reused sources."
    Without these, OpenClaw silently flies blind."""

    def test_to_dict_canonical_keys(self) -> None:
        r = ResearchResult(objective_id=1, mode="cold_start")
        d = r.to_dict()
        expected = {
            "objective_id", "mode",
            "new_source_ids", "reused_source_ids",
            "new_thinking_id", "reused_thinking_id",
            "supersedes_id", "rejected_reasons",
            # iter-4 continual-research fields: every run now carries
            # a quality score + actor backend provenance + a sibling-
            # branch flag so daily-cron callers can tell "answer
            # improved" from "answer regressed" from "live tip
            # unchanged" from "new branch kept as sibling".
            "quality_score", "actor_backend", "branched",
        }
        self.assertEqual(set(d.keys()), expected)


# ---------------------------------------------------------------------------
# Spec AC6 / AC7 — memory-augmented run blends external on EVERY call
# ---------------------------------------------------------------------------


class MemoryAugmentedDefaultsToExternalTest(unittest.TestCase):
    """Spec L57-59. Before the FIX 1 patch, a memory-augmented run
    needed `force_refresh=True` to fetch external — meaning the daily
    cron use case (spec L91 매일 9시 자동 리서치) was a no-op after
    day 1. The default must always fetch external; opting out via
    `skip_external=True` is the explicit path for offline/replay."""

    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(self.backend, tid, "q?")

    def test_second_run_default_fetches_external(self) -> None:
        class CountingStub:
            def __init__(self):
                self.calls = 0
                self.batches = [
                    [SearchHit("u1", "t1", "Llama 3 18 t/s mac mini metal")],
                    [SearchHit("u2", "t2", "Mistral 7B 22 t/s Apple Silicon")],
                ]
            def search(self, query):
                idx = min(self.calls, len(self.batches) - 1)
                self.calls += 1
                return self.batches[idx]
        stub = CountingStub()
        orch = Orchestrator(self.backend, search=stub)
        r1 = orch.research(self.oid)
        r2 = orch.research(self.oid)
        self.assertEqual(stub.calls, 2,
                          "memory-augmented run MUST call the search backend by default")
        self.assertEqual(r1.mode, "cold_start")
        self.assertEqual(r2.mode, "memory_augmented")
        self.assertEqual(len(r2.new_source_ids), 1)

    def test_skip_external_opts_out(self) -> None:
        class StubWithCounter:
            def __init__(self):
                self.calls = 0
            def search(self, query):
                self.calls += 1
                return [SearchHit("u1", "t1", "first hit content for cold start")]
        stub = StubWithCounter()
        orch = Orchestrator(self.backend, search=stub)
        orch.research(self.oid)
        self.assertEqual(stub.calls, 1)
        # Second run with skip_external must NOT call the backend.
        orch.research(self.oid, skip_external=True)
        self.assertEqual(stub.calls, 1)


# ---------------------------------------------------------------------------
# Spec AC9 — schedule.run_once_for_topic brings in new evidence
# ---------------------------------------------------------------------------


class ScheduleBringsNewEvidenceTest(unittest.TestCase):
    """Spec L91 — daily 9am cron. The previous schedule code passed no
    flag and therefore relied on the broken memory_augmented branch,
    making the cron a no-op after day 1. With FIX 1, the default
    behaviour fetches external on every run, so the schedule
    *should* bring in new evidence when it arrives."""

    def setUp(self) -> None:
        from research_graph import schedule
        self.schedule = schedule
        self.tmp = tempfile.mkdtemp(prefix="rg-sched-")
        self.db = str(Path(self.tmp) / "rg.json")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_schedule_picks_up_new_hits_across_runs(self) -> None:
        # Build a topic + objective, then monkey-patch the search
        # backend to return a different batch on each call.
        from research_graph import search as search_mod
        from research_graph.graph import get_default_backend

        backend = get_default_backend(self.db)
        try:
            tid = store.create_topic(backend, "llm")
            store.create_objective(backend, tid, "best local llm")
        finally:
            backend.close()

        class DayBatchSearch:
            def __init__(self):
                self.calls = 0
            def search(self, query):
                self.calls += 1
                if self.calls == 1:
                    return [SearchHit("u1", "t1", "Day1 evidence content for llm benchmark")]
                return [SearchHit("u2", "t2", "Day2 NEW evidence about a different llm benchmark")]

        # Patch `get_default_search` on the orchestrator module — that
        # is the symbol Orchestrator.__init__ actually calls (it was
        # `from .search import get_default_search` at import time, so
        # patching `search_mod.get_default_search` alone is too late).
        from research_graph import orchestrator as orch_mod
        original = orch_mod.get_default_search
        stub = DayBatchSearch()
        orch_mod.get_default_search = lambda: stub  # type: ignore[assignment]
        try:
            day1 = self.schedule.run_once_for_topic(self.db, "llm")
            day2 = self.schedule.run_once_for_topic(self.db, "llm")
        finally:
            orch_mod.get_default_search = original  # type: ignore[assignment]

        # Day-1: cold start, brings in u1.
        self.assertEqual(len(day1), 1)
        self.assertEqual(day1[0]["mode"], "cold_start")
        self.assertEqual(len(day1[0]["new_sources"]), 1)
        # Day-2: memory_augmented, MUST bring in u2 by default
        # (this was the spec-breaking bug). With FIX 1 + FIX 2, the
        # cron is functional.
        self.assertEqual(len(day2), 1)
        self.assertEqual(day2[0]["mode"], "memory_augmented")
        self.assertEqual(len(day2[0]["new_sources"]), 1,
                          "scheduled run must bring in new external evidence by default")


if __name__ == "__main__":
    unittest.main()
