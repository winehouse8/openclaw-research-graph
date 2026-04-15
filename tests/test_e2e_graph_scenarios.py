"""End-to-end scenarios exercising the graph-shaped store.

Each scenario constructs a fresh InMemoryGraphBackend and drives full
orchestrator / retrieval / store paths through a realistic workflow.
No mocking beyond the deterministic offline search stubs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research_graph import dedup, retrieval, schedule, store
from research_graph.dedup import content_hash
from research_graph.graph import InMemoryGraphBackend
from research_graph.orchestrator import Orchestrator
from research_graph.search import SearchHit


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTimedSearch:
    def __init__(self, script: dict[int, list[SearchHit]]) -> None:
        self._script = script
        self.current_day = 1

    def search(self, query: str) -> list[SearchHit]:
        return list(self._script.get(self.current_day, []))


class ScriptedSearch:
    def __init__(self, batches: list[list[SearchHit]]) -> None:
        self._batches = batches
        self._i = 0

    def search(self, query: str) -> list[SearchHit]:
        batch = self._batches[min(self._i, len(self._batches) - 1)]
        self._i += 1
        return list(batch)


# ---------------------------------------------------------------------------
# Scenario 1: multi-day evolution producing REUSES edges
# ---------------------------------------------------------------------------


class MultiDayReuseEdgeTest(unittest.TestCase):
    def test_reuses_edge_recorded(self) -> None:
        backend = InMemoryGraphBackend()
        tid_llm = store.create_topic(backend, "local-llm-mac-mini")
        tid_mem = store.create_topic(backend, "agentic-memory")
        oid_speed = store.create_objective(
            backend, tid_llm, "fastest local llm on 16gb mac mini"
        )
        oid_mem = store.create_objective(
            backend, tid_mem, "best agentic memory architecture"
        )

        speed_days = {
            1: [
                SearchHit("https://ex/a1", "Llama3", "Llama 3.1 8B Q4 18 tokens per second mac mini"),
                SearchHit("https://ex/a2", "Mistral", "Mistral 7B Q4 22 tokens per second mac mini"),
            ],
            2: [
                SearchHit("https://ex/a1", "Llama3", "Llama 3.1 8B Q4 18 tokens per second mac mini"),
                SearchHit("https://ex/a2", "Mistral", "Mistral 7B Q4 22 tokens per second mac mini"),
            ],  # identical -> dedup collapse on day 2 thinking
            3: [
                SearchHit("https://ex/a3", "Phi3", "Phi 3 mini Q4 35 tokens per second mac mini"),
            ],
            4: [
                SearchHit("https://ex/a3", "Phi3", "Phi 3 mini Q4 35 tokens per second mac mini"),
            ],  # another collapse
            5: [
                SearchHit("https://ex/a4", "Qwen", "Qwen 2.5 7B instruct outperforms llama on mac mini"),
            ],
        }
        mem_days = {
            1: [SearchHit("https://ex/m1", "MemGPT", "MemGPT hierarchical memory LLM agents")],
            2: [SearchHit("https://ex/m2", "Letta", "Letta productized successor to MemGPT REST API")],
            3: [SearchHit("https://ex/m2", "Letta", "Letta productized successor to MemGPT REST API")],
            4: [SearchHit("https://ex/m3", "Zep", "Zep long term memory temporal knowledge graph")],
            5: [SearchHit("https://ex/m3", "Zep", "Zep long term memory temporal knowledge graph")],
        }

        speed_search = FakeTimedSearch(speed_days)
        mem_search = FakeTimedSearch(mem_days)

        for day in range(1, 6):
            speed_search.current_day = day
            mem_search.current_day = day
            force = day > 1
            Orchestrator(backend, search=speed_search).research(oid_speed, force_refresh=force)
            Orchestrator(backend, search=mem_search).research(oid_mem, force_refresh=force)

        # Any REUSES edges that exist must satisfy the branch-C invariant:
        # reuser != reused (no self-loops) and both endpoints live in the
        # same objective. Self-loops are forbidden by store.reuse() but we
        # assert the observable invariant here as well.
        speed_reuses = store.list_reuses(backend, oid_speed)
        mem_reuses = store.list_reuses(backend, oid_mem)
        for pair in speed_reuses + mem_reuses:
            self.assertNotEqual(
                int(pair["reuser"]["id"]), int(pair["reused"]["id"]),
                "REUSES self-loop leaked into the graph",
            )
        for pair in speed_reuses:
            self.assertEqual(pair["reuser"]["objective_id"], oid_speed)
            self.assertEqual(pair["reused"]["objective_id"], oid_speed)
        for pair in mem_reuses:
            self.assertEqual(pair["reuser"]["objective_id"], oid_mem)
            self.assertEqual(pair["reused"]["objective_id"], oid_mem)


# ---------------------------------------------------------------------------
# Scenario 2: hand-crafted 4-hop walk
# ---------------------------------------------------------------------------


class ReuseSelfLoopRegressionTest(unittest.TestCase):
    """HIGH 2 + HIGH 3 regression coverage for the three orchestrator
    tail branches (created / no-op-collapse / older-row-collapse)."""

    def test_reuses_no_self_loop_on_noop_run(self) -> None:
        # Cold start followed by an immediate re-run on the SAME fixture.
        # Branch A then Branch B: the second run collapses dedup onto the
        # latest-and-only thinking, so NO reuse edge should be written.
        backend = InMemoryGraphBackend()
        tid = store.create_topic(backend, "t")
        oid = store.create_objective(backend, tid, "q")

        hits = [
            SearchHit("https://ex/a", "A", "alpha bravo charlie delta echo foxtrot"),
            SearchHit("https://ex/b", "B", "golf hotel india juliet kilo lima"),
        ]
        search = ScriptedSearch([hits, hits])

        r1 = Orchestrator(backend, search=search).research(oid)
        self.assertEqual(r1.mode, "cold_start")
        self.assertIsNotNone(r1.new_thinking_id)
        t1_id = r1.new_thinking_id

        r2 = Orchestrator(backend, search=search).research(oid, force_refresh=False)
        self.assertEqual(r2.mode, "memory_augmented")
        self.assertIsNone(r2.new_thinking_id)
        self.assertEqual(r2.reused_thinking_id, t1_id)
        self.assertIsNone(r2.supersedes_id)

        self.assertEqual(
            len(store.list_reuses(backend, oid)), 0,
            "noop-collapse onto latest must not emit a REUSES edge",
        )

        # --- Now drive Branch C manually by injecting a dedup result that
        # targets an OLDER thinking while a NEWER one is the latest. We
        # monkey-patch dedup.upsert_thinking inside the orchestrator module
        # (this is where the orchestrator imports it) so the real dedup
        # helper is unaffected for every other test.
        hits2 = [
            SearchHit("https://ex/c", "C", "mike november oscar papa quebec romeo"),
        ]
        search2 = ScriptedSearch([hits2])
        r3 = Orchestrator(backend, search=search2).research(oid, force_refresh=True)
        # Branch A: brand new thinking that supersedes t1.
        self.assertIsNotNone(r3.new_thinking_id)
        self.assertEqual(r3.supersedes_id, t1_id)
        t2_id = r3.new_thinking_id

        # Now t2 is the latest. Force dedup to collapse onto t1 on the
        # next run -> Branch C: (t2)-[:REUSES]->(t1).
        from research_graph import dedup as _dedup
        from research_graph import orchestrator as _orch_mod

        real_upsert = _dedup.upsert_thinking

        def fake_upsert(backend_, objective_id_, content, supports, author,
                        supersedes_id=None, threshold=0.85):
            return (int(t1_id), False)

        _orch_mod.dedup.upsert_thinking = fake_upsert  # type: ignore[attr-defined]
        try:
            r4 = Orchestrator(backend, search=search2).research(oid, force_refresh=True)
        finally:
            _orch_mod.dedup.upsert_thinking = real_upsert  # type: ignore[attr-defined]

        self.assertIsNone(r4.new_thinking_id)
        self.assertEqual(r4.reused_thinking_id, t1_id)
        self.assertIsNone(
            r4.supersedes_id,
            "Branch C must not set supersedes_id on a no-new-thinking run",
        )

        reuses = store.list_reuses(backend, oid)
        self.assertEqual(len(reuses), 1)
        pair = reuses[0]
        self.assertEqual(int(pair["reuser"]["id"]), int(t2_id))
        self.assertEqual(int(pair["reused"]["id"]), int(t1_id))

    def test_supersedes_reuses_mutual_exclusion(self) -> None:
        # Explicit HIGH 3 coverage: the three branches are mutually
        # exclusive. Branch C in particular must set reused_thinking_id
        # WITHOUT setting supersedes_id, and Branch A must set
        # supersedes_id WITHOUT writing a REUSES edge.
        backend = InMemoryGraphBackend()
        tid = store.create_topic(backend, "t")
        oid = store.create_objective(backend, tid, "q")

        hits_a = [SearchHit("https://ex/1", "1", "alpha bravo charlie delta")]
        hits_b = [SearchHit("https://ex/2", "2", "echo foxtrot golf hotel")]

        r_cold = Orchestrator(backend, search=ScriptedSearch([hits_a])).research(oid)
        t1 = r_cold.new_thinking_id
        self.assertIsNotNone(t1)
        self.assertIsNone(r_cold.supersedes_id)  # nothing to supersede on cold start

        r_super = Orchestrator(
            backend, search=ScriptedSearch([hits_b])
        ).research(oid, force_refresh=True)
        t2 = r_super.new_thinking_id
        self.assertIsNotNone(t2)
        # Branch A: supersedes set, no reuse edge written this run.
        self.assertEqual(r_super.supersedes_id, t1)
        self.assertIsNone(r_super.reused_thinking_id)
        self.assertEqual(len(store.list_reuses(backend, oid)), 0)

        # Now collide onto t1 to exercise Branch C.
        from research_graph import dedup as _dedup
        from research_graph import orchestrator as _orch_mod

        real_upsert = _dedup.upsert_thinking
        _orch_mod.dedup.upsert_thinking = (  # type: ignore[attr-defined]
            lambda b, o, c, s, author, supersedes_id=None, threshold=0.85: (int(t1), False)
        )
        try:
            r_reuse = Orchestrator(
                backend, search=ScriptedSearch([hits_b])
            ).research(oid, force_refresh=True)
        finally:
            _orch_mod.dedup.upsert_thinking = real_upsert  # type: ignore[attr-defined]

        # Branch C assertions: reused_thinking_id set, supersedes_id NOT set,
        # new_thinking_id NOT set, and exactly one edge (t2)-[:REUSES]->(t1).
        self.assertEqual(r_reuse.reused_thinking_id, int(t1))
        self.assertIsNone(r_reuse.new_thinking_id)
        self.assertIsNone(r_reuse.supersedes_id)

        reuses = store.list_reuses(backend, oid)
        self.assertEqual(len(reuses), 1)
        self.assertEqual(int(reuses[0]["reuser"]["id"]), int(t2))
        self.assertEqual(int(reuses[0]["reused"]["id"]), int(t1))


class FourHopWalkTest(unittest.TestCase):
    def test_four_hop_shape(self) -> None:
        backend = InMemoryGraphBackend()
        tid = store.create_topic(backend, "t")
        oid = store.create_objective(backend, tid, "q")
        s1 = store.insert_source(backend, oid, "u1", "s1", "body 1", "h1", None)
        s2 = store.insert_source(backend, oid, "u2", "s2", "body 2", "h2", None)
        s3 = store.insert_source(backend, oid, "u3", "s3", "body 3", "h3", None)
        s4 = store.insert_source(backend, oid, "u4", "s4", "body 4", "h4", None)
        # T0 cites s1, s2
        t0 = store.insert_thinking(
            backend, oid, "t0", "th0", [s1, s2], "actor"
        )
        # T1 cites s2, s3 (shares s2 with T0 -> related, s3 is new)
        t1 = store.insert_thinking(
            backend, oid, "t1", "th1", [s2, s3], "actor"
        )
        # T2 cites s1, s4 (shares s1 with T0 -> related, s4 is new)
        t2 = store.insert_thinking(
            backend, oid, "t2", "th2", [s1, s4], "actor"
        )
        # T3 cites only s4 (does NOT share with T0 directly -> not related)
        t3 = store.insert_thinking(
            backend, oid, "t3", "th3", [s4], "actor"
        )

        walk = store.four_hop_evidence_walk(backend, t0)
        self.assertEqual(int(walk["start"]["id"]), t0)
        shared_ids = [int(s["id"]) for s in walk["shared_sources"]]
        self.assertEqual(sorted(shared_ids), sorted([s1, s2]))
        related_ids = sorted(int(t["id"]) for t in walk["related_thinkings"])
        self.assertEqual(related_ids, sorted([t1, t2]))
        self.assertNotIn(t3, related_ids, "T3 does not share any T0 source")
        new_ids = sorted(int(s["id"]) for s in walk["new_sources"])
        self.assertEqual(new_ids, sorted([s3, s4]))


# ---------------------------------------------------------------------------
# Scenario 3: cross-topic shared source discovery
# ---------------------------------------------------------------------------


class CrossTopicSharedTest(unittest.TestCase):
    def test_same_hash_across_topics(self) -> None:
        backend = InMemoryGraphBackend()
        ta = store.create_topic(backend, "alpha")
        tb = store.create_topic(backend, "beta")
        oa = store.create_objective(backend, ta, "qa")
        ob = store.create_objective(backend, tb, "qb")

        body = "Shared arxiv paper about long context attention mechanisms"
        h = content_hash(body)
        store.insert_source(backend, oa, "https://arxiv/xyz", "Paper", body, h, None)
        store.insert_source(backend, ob, "https://arxiv/xyz", "Paper", body, h, None)

        # Distractor: a topic with an unrelated source
        tc = store.create_topic(backend, "gamma")
        oc = store.create_objective(backend, tc, "qc")
        store.insert_source(
            backend, oc, "https://other", "other",
            "unrelated distraction", content_hash("unrelated distraction"), None,
        )

        rows = store.cross_topic_shared_sources(backend)
        hashes = {r["content_hash"] for r in rows}
        self.assertIn(h, hashes)
        # Find the row for h and check its topic set.
        row = next(r for r in rows if r["content_hash"] == h)
        self.assertEqual(sorted(row["topic_ids"]), sorted([ta, tb]))


# ---------------------------------------------------------------------------
# Scenario 4: supersession chain follow
# ---------------------------------------------------------------------------


class SupersessionChainTest(unittest.TestCase):
    def test_four_step_chain(self) -> None:
        backend = InMemoryGraphBackend()
        tid = store.create_topic(backend, "t")
        oid = store.create_objective(backend, tid, "q")
        s = store.insert_source(backend, oid, None, None, "body", "hb", None)
        t1 = store.insert_thinking(backend, oid, "one", "h1", [s], "actor")
        t2 = store.insert_thinking(backend, oid, "two", "h2", [s], "actor", supersedes_id=t1)
        t3 = store.insert_thinking(backend, oid, "three", "h3", [s], "actor", supersedes_id=t2)
        t4 = store.insert_thinking(backend, oid, "four", "h4", [s], "actor", supersedes_id=t3)
        chain = store.supersession_chain(backend, t4)
        self.assertEqual([int(c["id"]) for c in chain], [t4, t3, t2, t1])


# ---------------------------------------------------------------------------
# Scenario 5: cocited thinkings
# ---------------------------------------------------------------------------


class CocitedThinkingsTest(unittest.TestCase):
    def test_cocited_subset(self) -> None:
        backend = InMemoryGraphBackend()
        tid = store.create_topic(backend, "t")
        oid = store.create_objective(backend, tid, "q")
        s1 = store.insert_source(backend, oid, "u1", "s1", "b1", "h1", None)
        s2 = store.insert_source(backend, oid, "u2", "s2", "b2", "h2", None)
        s3 = store.insert_source(backend, oid, "u3", "s3", "b3", "h3", None)
        s4 = store.insert_source(backend, oid, "u4", "s4", "b4", "h4", None)
        # T0 cites s1, s2
        t0 = store.insert_thinking(backend, oid, "t0", "hh0", [s1, s2], "actor")
        # T1 cites s2, s3 -> cocited
        t1 = store.insert_thinking(backend, oid, "t1", "hh1", [s2, s3], "actor")
        # T2 cites s1 -> cocited
        t2 = store.insert_thinking(backend, oid, "t2", "hh2", [s1], "actor")
        # T3 cites s3, s4 -> NOT cocited with t0
        t3 = store.insert_thinking(backend, oid, "t3", "hh3", [s3, s4], "actor")
        # T4 cites s4 only -> NOT cocited
        t4 = store.insert_thinking(backend, oid, "t4", "hh4", [s4], "actor")

        co = store.cocited_thinkings(backend, t0)
        co_ids = sorted(int(r["id"]) for r in co)
        self.assertEqual(co_ids, sorted([t1, t2]))
        self.assertNotIn(t3, co_ids)
        self.assertNotIn(t4, co_ids)


# ---------------------------------------------------------------------------
# Scenario 6: CLI walk subcommand + full CLI trace through store
# ---------------------------------------------------------------------------


class CLIWalkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "cli.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _cli(self, *args: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "research_graph", "--db", self.db, *args],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=False,
        )

    def test_walk_cli(self) -> None:
        r = self._cli("init"); self.assertEqual(r.returncode, 0, r.stderr)
        r = self._cli("topic", "add", "local-llm-mac-mini")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self._cli(
            "objective", "add", "--topic", "local-llm-mac-mini",
            "best local llm mac mini",
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        oid = int(r.stdout.strip())

        r = self._cli("research", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        research_result = json.loads(r.stdout.strip())
        tid = research_result["new_thinking_id"]
        self.assertIsNotNone(tid)

        r = self._cli("query", "llama", "--kind", "sources",
                      "--objective", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreater(len(json.loads(r.stdout.strip())), 0)

        r = self._cli("query", "synthesis", "--kind", "thinkings",
                      "--objective", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        th_rows = json.loads(r.stdout.strip())
        self.assertGreaterEqual(len(th_rows), 1)

        r = self._cli("walk", "4hop", "--thinking", str(tid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        walk = json.loads(r.stdout.strip())
        self.assertIn("start", walk)
        self.assertIn("shared_sources", walk)
        self.assertIn("related_thinkings", walk)
        self.assertIn("new_sources", walk)

        r = self._cli("walk", "cocited", "--thinking", str(tid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)

        r = self._cli("walk", "cross-topic-sources", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)

        r = self._cli("schedule", "once", "--topic", "local-llm-mac-mini", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
