"""Heavyweight end-to-end integration scenarios for research_graph.

Each test creates a fresh tempdir DB and walks a full multi-step workflow
through the real api / Orchestrator / storage / CLI surfaces with no
mocking. These are deliberately "not simple unit tests" -- they cross
module boundaries, simulate time, adversarial actors, and realistic
OpenClaw orchestrator cadence.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research_graph import api, dedup, reasoning, retrieval, schedule, storage
from research_graph.orchestrator import Orchestrator
from research_graph.search import (
    ExternalSearch,
    SearchHit,
    _BUILTIN_BACKENDS,
    register_backend,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class FakeTimedSearch:
    """Search backend whose results depend on an externally-bumped ``current_day``.

    Used by the multi-day evolution tests so we can drive the orchestrator
    through simulated time without actually sleeping.
    """

    def __init__(self, day_to_hits: dict[int, list[SearchHit]]) -> None:
        self._days = day_to_hits
        self.current_day = 1
        self.calls: list[tuple[int, str]] = []

    def search(self, query: str) -> list[SearchHit]:
        self.calls.append((self.current_day, query))
        return list(self._days.get(self.current_day, []))


class ScriptedSearch:
    """Returns a distinct batch per call, cycling through a list."""

    def __init__(self, batches: list[list[SearchHit]]) -> None:
        self._batches = batches
        self._i = 0

    def search(self, query: str) -> list[SearchHit]:
        batch = self._batches[min(self._i, len(self._batches) - 1)]
        self._i += 1
        return list(batch)


# ---------------------------------------------------------------------------
# E2E-1: multi-topic multi-objective evolution over a simulated week
# ---------------------------------------------------------------------------


class MultiTopicEvolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "week.db"
        self.conn = storage.connect(self.db)
        self.llm_topic = storage.create_topic(self.conn, "local-llm-mac-mini")
        self.mem_topic = storage.create_topic(self.conn, "agentic-memory")
        self.llm_obj_speed = storage.create_objective(
            self.conn, self.llm_topic, "fastest local llm on 16gb mac mini"
        )
        self.llm_obj_quality = storage.create_objective(
            self.conn, self.llm_topic, "best instruction following local llm mac mini"
        )
        self.mem_obj = storage.create_objective(
            self.conn, self.mem_topic, "best agentic memory architecture"
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_week_long_evolution(self) -> None:
        # Day-indexed search batches per objective. We simulate:
        #   day 1: initial corpus
        #   day 2: one new source, one overlap
        #   day 3: contradicting update that should force a new thinking
        #   day 4: exact repeat of day 3 -> dedup collapses, reuse reported
        #   day 5: partial overlap, one new source
        llm_speed_days: dict[int, list[SearchHit]] = {
            1: [
                SearchHit("https://ex/a1", "Llama3 bench",
                          "Llama 3.1 8B Q4 runs at 18 tokens per second on Mac mini metal"),
                SearchHit("https://ex/a2", "Mistral bench",
                          "Mistral 7B Q4 reaches 22 tokens per second on Apple Silicon mac mini"),
            ],
            2: [
                SearchHit("https://ex/a1", "Llama3 bench",
                          "Llama 3.1 8B Q4 runs at 18 tokens per second on Mac mini metal"),
                SearchHit("https://ex/a3", "Phi3 bench",
                          "Phi 3 mini Q4 runs at 35 tokens per second comfortably on 16gb mac mini"),
            ],
            3: [
                # contradiction: new 2026 ranking supersedes earlier claim
                SearchHit("https://ex/a4", "2026 ranking",
                          "Qwen 2.5 7B instruct Q4_K_M now outperforms Llama 3.1 on mac mini benchmarks 2026"),
            ],
            4: [
                # exact repeat of day 3
                SearchHit("https://ex/a4", "2026 ranking",
                          "Qwen 2.5 7B instruct Q4_K_M now outperforms Llama 3.1 on mac mini benchmarks 2026"),
            ],
            5: [
                # partial overlap with day 3 plus a fresh one
                SearchHit("https://ex/a4", "2026 ranking",
                          "Qwen 2.5 7B instruct Q4_K_M now outperforms Llama 3.1 on mac mini benchmarks 2026"),
                SearchHit("https://ex/a5", "Gemma2 bench",
                          "Gemma 2 9B quantized runs at 15 tokens per second on 16gb mac mini metal"),
            ],
        }
        llm_quality_days: dict[int, list[SearchHit]] = {
            1: [
                SearchHit("https://ex/q1", "instruct llama",
                          "Llama 3.1 8B instruct follows multi-step instructions on mac mini"),
            ],
            2: [
                SearchHit("https://ex/q2", "instruct qwen",
                          "Qwen 2.5 7B instruct answers multi-step questions accurately"),
            ],
            3: [
                SearchHit("https://ex/q2", "instruct qwen",
                          "Qwen 2.5 7B instruct answers multi-step questions accurately"),
            ],
            4: [
                SearchHit("https://ex/q3", "instruct phi",
                          "Phi 3 mini instruct handles short instructions but struggles with long context"),
            ],
            5: [
                SearchHit("https://ex/q3", "instruct phi",
                          "Phi 3 mini instruct handles short instructions but struggles with long context"),
            ],
        }
        mem_days: dict[int, list[SearchHit]] = {
            1: [
                SearchHit("https://ex/m1", "memgpt",
                          "MemGPT hierarchical memory for LLM agents with working set and archival store"),
            ],
            2: [
                SearchHit("https://ex/m2", "letta",
                          "Letta is the productized successor to MemGPT exposing a REST API"),
            ],
            3: [
                SearchHit("https://ex/m2", "letta",
                          "Letta is the productized successor to MemGPT exposing a REST API"),
            ],
            4: [
                SearchHit("https://ex/m2", "letta",
                          "Letta is the productized successor to MemGPT exposing a REST API"),
            ],
            5: [
                SearchHit("https://ex/m3", "zep",
                          "Zep is a long term memory store for LLM agents with a temporal knowledge graph"),
            ],
        }

        llm_speed_search = FakeTimedSearch(llm_speed_days)
        llm_quality_search = FakeTimedSearch(llm_quality_days)
        mem_search = FakeTimedSearch(mem_days)

        def orch_for(backend: FakeTimedSearch) -> Orchestrator:
            return Orchestrator(self.conn, search=backend)

        results_by_day: dict[int, dict[int, object]] = {}
        for day in range(1, 6):
            llm_speed_search.current_day = day
            llm_quality_search.current_day = day
            mem_search.current_day = day
            # The orchestrator re-uses the same search instance per objective
            # on day 1 (cold start); on subsequent days we force_refresh so
            # external search fires again.
            force = day > 1
            r_speed = orch_for(llm_speed_search).research(
                self.llm_obj_speed, force_refresh=force
            )
            r_quality = orch_for(llm_quality_search).research(
                self.llm_obj_quality, force_refresh=force
            )
            r_mem = orch_for(mem_search).research(
                self.mem_obj, force_refresh=force
            )
            results_by_day[day] = {
                self.llm_obj_speed: r_speed,
                self.llm_obj_quality: r_quality,
                self.mem_obj: r_mem,
            }

        # --- assertions --------------------------------------------------

        # Sources grew over time without exact-hash re-insertion.
        all_speed_sources = storage.list_sources(self.conn, self.llm_obj_speed)
        # day1: 2, day2: +1 (a3), day3: +1 (a4), day4: 0, day5: +1 (a5) = 5
        self.assertEqual(len(all_speed_sources), 5,
                         f"expected 5 distinct speed sources, got {len(all_speed_sources)}")
        urls = sorted({s["url"] for s in all_speed_sources})
        self.assertEqual(urls,
                         ["https://ex/a1", "https://ex/a2",
                          "https://ex/a3", "https://ex/a4", "https://ex/a5"])

        # Thinkings chain evolved.
        speed_thinkings = storage.list_thinkings(self.conn, self.llm_obj_speed)
        # day1 wrote one, day2 wrote one (new source a3), day3 wrote one (a4),
        # day4 no new sources -> dedup-reused, day5 wrote one (a5). => 4 rows.
        self.assertEqual(len(speed_thinkings), 4,
                         f"expected 4 speed thinkings, got {len(speed_thinkings)}")
        # day2 supersedes day1, day3 supersedes day2, day5 supersedes day3
        self.assertIsNone(speed_thinkings[0]["supersedes_id"])
        self.assertEqual(speed_thinkings[1]["supersedes_id"], speed_thinkings[0]["id"])
        self.assertEqual(speed_thinkings[2]["supersedes_id"], speed_thinkings[1]["id"])
        self.assertEqual(speed_thinkings[3]["supersedes_id"], speed_thinkings[2]["id"])

        # Day 4 must have reused_thinking_id set.
        day4_speed = results_by_day[4][self.llm_obj_speed]
        self.assertIsNone(day4_speed.new_thinking_id)
        self.assertIsNotNone(day4_speed.reused_thinking_id)
        self.assertEqual(day4_speed.reused_thinking_id, speed_thinkings[2]["id"])

        # Retrieval scoping: topic / objective / global slices
        speed_only = retrieval.search_sources(
            self.conn, "tokens per second mac mini",
            objective_id=self.llm_obj_speed, top_k=10, min_score=0.0,
        )
        self.assertTrue(all(s["objective_id"] == self.llm_obj_speed for s in speed_only))
        self.assertGreaterEqual(len(speed_only), 3)

        topic_wide = retrieval.search_sources(
            self.conn, "instruct mac mini",
            topic_id=self.llm_topic, top_k=20, min_score=0.0,
        )
        topic_obj_ids = {s["objective_id"] for s in topic_wide}
        self.assertTrue(topic_obj_ids.issubset({self.llm_obj_speed, self.llm_obj_quality}),
                        f"topic_wide leaked into other topics: {topic_obj_ids}")

        global_mem = retrieval.search_sources(
            self.conn, "memgpt letta memory agents",
            top_k=20, min_score=0.0,
        )
        mem_urls = {s["url"] for s in global_mem}
        self.assertIn("https://ex/m1", mem_urls)
        self.assertIn("https://ex/m2", mem_urls)

        # Cross-contamination check: a query scoped to local-llm-mac-mini
        # topic must never return agentic-memory sources.
        llm_topic_hits = retrieval.search_sources(
            self.conn, "memgpt letta agents",
            topic_id=self.llm_topic, top_k=20, min_score=0.0,
        )
        for row in llm_topic_hits:
            self.assertNotIn(row["url"], {"https://ex/m1", "https://ex/m2", "https://ex/m3"},
                             f"cross-topic leak: {row['url']}")


# ---------------------------------------------------------------------------
# E2E-2: adversarial actor smuggling raw source content
# ---------------------------------------------------------------------------


class AdversarialActorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = storage.connect(Path(self.tmp.name) / "adv.db")
        tid = storage.create_topic(self.conn, "adv")
        self.oid = storage.create_objective(self.conn, tid, "adversarial test")
        self.long_passage = (
            "The Apple M2 Pro chip inside the 2023 mac mini delivers sustained "
            "memory bandwidth that allows quantized large language models up to "
            "about 13 billion parameters to run comfortably at interactive latency "
            "provided the working set stays resident in unified memory and the "
            "model is loaded with metal acceleration enabled end-to-end pipeline"
        )
        # sanity: ensure the passage is ~250 chars informative text
        self.assertGreaterEqual(len(self.long_passage), 250)
        self.sid = dedup.upsert_source(
            self.conn, self.oid, "https://ex/adv", "adv", self.long_passage,
        )[0]

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_offset_smuggle_rejected(self) -> None:
        # Embed verbatim starting at source offset 17, NOT at offset 0,
        # with 30 chars of decoy in front to fool naive startswith checks.
        decoy = "x" * 30
        smuggled = self.long_passage[17:17 + 210]  # 210 > MAX_QUOTE_LEN (200)
        adversarial = f"{decoy}\n\nKey finding: {smuggled}"
        verdict = reasoning.critic_verify(self.conn, adversarial, [self.sid])
        self.assertFalse(verdict.accepted,
                         f"adversarial quote should be rejected; reasons={verdict.reasons}")
        self.assertTrue(any("quoted verbatim" in r for r in verdict.reasons))

    def test_well_behaved_paraphrase_accepted(self) -> None:
        # Only ~30 chars verbatim from an arbitrary offset, rest paraphrased.
        short_quote = self.long_passage[45:45 + 30]
        paraphrase = (
            "Synthesis for adversarial test: the passage claims that mid-sized "
            "quantized models fit in unified memory when loaded with metal. "
            f"It notes \"{short_quote}\" as a supporting observation."
        )
        verdict = reasoning.critic_verify(self.conn, paraphrase, [self.sid])
        self.assertTrue(verdict.accepted,
                        f"paraphrase should be accepted; reasons={verdict.reasons}")


# ---------------------------------------------------------------------------
# E2E-3: keyword search variation
# ---------------------------------------------------------------------------


class KeywordSearchVariationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = storage.connect(Path(self.tmp.name) / "kw.db")
        tid = storage.create_topic(self.conn, "kw")
        self.oid = storage.create_objective(self.conn, tid, "mac mini local llm benchmarks")
        corpus = [
            "Apple Silicon M1 mac mini runs quantized Llama models at interactive speed with metal acceleration",
            "M2 Pro mac mini delivers high memory bandwidth suitable for 13 billion parameter local llm workloads",
            "Quantization to Q4_K_M reduces local llm memory footprint while preserving most instruction following quality",
            "Ollama wraps llama.cpp and provides a simple command line for running local llm on mac mini and other unix systems",
            "The Mistral 7B Q4 model reaches around 22 tokens per second on a 16gb mac mini in benchmarks",
            "Phi 3 mini is a small local llm from Microsoft that runs comfortably on 8gb memory laptops and desktops",
            "Qwen 2.5 7B instruct is a recent local llm strong on multi step reasoning benchmarks",
            "Gemma 2 9B quantized runs at roughly 15 tokens per second on the mac mini with metal backend",
            "Llama 3.1 8B Q4 is a common local llm baseline and runs at about 18 tokens per second on mac mini hardware",
            "Chocolate chip cookie recipes call for butter sugar flour eggs and baking soda and have nothing to do with computers",
        ]
        for text in corpus:
            dedup.upsert_source(self.conn, self.oid, None, None, text)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_exact_keyword_query_finds_relevant_sources(self) -> None:
        # Exact-keyword match: expect llm/mac/mini docs at the top.
        rows = retrieval.search_sources(
            self.conn, "mac mini local llm benchmarks",
            objective_id=self.oid, top_k=5, min_score=0.05,
        )
        self.assertGreaterEqual(len(rows), 3)
        joined = " ".join(r["content"] for r in rows).lower()
        self.assertIn("mac mini", joined)
        self.assertNotIn("cookie", joined,
                         "cookie recipe must not surface in exact-keyword query")

    def test_related_vocabulary_query(self) -> None:
        # Related query uses vocabulary overlap but different surface words.
        rows = retrieval.search_sources(
            self.conn, "apple silicon quantized model tokens per second",
            objective_id=self.oid, top_k=5, min_score=0.05,
        )
        self.assertGreaterEqual(len(rows), 2)
        for r in rows:
            self.assertNotIn("cookie", r["content"].lower())

    def test_unrelated_query_returns_nothing_above_threshold(self) -> None:
        # A totally unrelated query with no overlapping content terms
        # must fall below the 0.05 min_score threshold.
        rows = retrieval.search_sources(
            self.conn, "submarine sonar dolphin echolocation biology",
            objective_id=self.oid, top_k=5, min_score=0.05,
        )
        self.assertEqual(rows, [],
                         f"unrelated query leaked results: {[r['content'][:40] for r in rows]}")

    def test_keyword_ranking_is_sensible(self) -> None:
        # The Llama-specific query should put a Llama row ahead of the Phi row.
        rows = retrieval.search_sources(
            self.conn, "llama mac mini tokens per second",
            objective_id=self.oid, top_k=10, min_score=0.0,
        )
        self.assertGreater(len(rows), 0)
        top_content = rows[0]["content"].lower()
        self.assertIn("llama", top_content)


# ---------------------------------------------------------------------------
# E2E-4: supersession chain integrity across dedup-collapse days
# ---------------------------------------------------------------------------


class SupersessionChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = storage.connect(Path(self.tmp.name) / "chain.db")
        tid = storage.create_topic(self.conn, "chain")
        self.oid = storage.create_objective(self.conn, tid, "supersession question")

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_chain_handles_dedup_collapse(self) -> None:
        # Day 1: fresh facts
        day1 = [SearchHit("https://ex/d1a", "d1a", "alpha acceleration benchmark mac mini metal"),
                SearchHit("https://ex/d1b", "d1b", "beta baseline latency metal mac mini")]
        # Day 2: new fact arrives
        day2 = day1 + [SearchHit("https://ex/d2", "d2", "gamma improvement memory bandwidth mac mini")]
        # Day 3: identical to day 2 -> no new sources -> thinking text
        # identical -> dedup collapses
        day3 = list(day2)
        # Day 4: additional new fact -> new thinking; must supersede day-2 thinking
        # (because day-3 did NOT create a new thinking row)
        day4 = day2 + [SearchHit("https://ex/d4", "d4", "delta new evaluation mac mini results")]

        sequences = [day1, day2, day3, day4]
        scripted = ScriptedSearch(sequences)
        orch = Orchestrator(self.conn, search=scripted)

        r1 = orch.research(self.oid)
        self.assertEqual(r1.mode, "cold_start")
        self.assertIsNotNone(r1.new_thinking_id)
        self.assertIsNone(r1.supersedes_id)

        r2 = orch.research(self.oid, force_refresh=True)
        self.assertIsNotNone(r2.new_thinking_id)
        self.assertEqual(r2.supersedes_id, r1.new_thinking_id)

        r3 = orch.research(self.oid, force_refresh=True)
        # day3 brings no new sources -> actor output identical -> dedup collapse
        self.assertIsNone(r3.new_thinking_id)
        self.assertEqual(r3.reused_thinking_id, r2.new_thinking_id)

        r4 = orch.research(self.oid, force_refresh=True)
        self.assertIsNotNone(r4.new_thinking_id)
        # day4's new thinking must chain from the latest NON-collapsed row,
        # which is r2 (day2), because r3 never inserted a row.
        self.assertEqual(r4.supersedes_id, r2.new_thinking_id)

        # No thinking was ever deleted: rows 1, 2, 4 all present.
        all_thinkings = storage.list_thinkings(self.conn, self.oid)
        ids = [t["id"] for t in all_thinkings]
        self.assertIn(r1.new_thinking_id, ids)
        self.assertIn(r2.new_thinking_id, ids)
        self.assertIn(r4.new_thinking_id, ids)
        self.assertEqual(len(all_thinkings), 3)


# ---------------------------------------------------------------------------
# E2E-5: CLI end-to-end subprocess trace
# ---------------------------------------------------------------------------


class CLIEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "cli.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _cli(self, *args: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "research_graph", "--db", self.db, *args],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=False,
        )

    def test_full_cli_trace(self) -> None:
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
        d = json.loads(r.stdout.strip())
        expected_keys = {
            "objective_id", "mode", "new_source_ids",
            "new_thinking_id", "reused_thinking_id", "supersedes_id",
        }
        self.assertEqual(set(d.keys()), expected_keys)
        self.assertEqual(d["mode"], "cold_start")
        self.assertGreater(len(d["new_source_ids"]), 0)
        self.assertIsNotNone(d["new_thinking_id"])

        # Query kind separation: sources vs thinkings should not cross.
        r = self._cli("query", "llama mac mini", "--kind", "sources",
                      "--objective", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        src_rows = json.loads(r.stdout.strip())
        self.assertGreater(len(src_rows), 0)
        for row in src_rows:
            self.assertIn("url", row)
            self.assertIn("content_hash", row)
            self.assertNotIn("supports_source_ids", row)

        r = self._cli("query", "synthesis", "--kind", "thinkings",
                      "--objective", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        th_rows = json.loads(r.stdout.strip())
        self.assertGreaterEqual(len(th_rows), 1)
        for row in th_rows:
            self.assertIn("supports_source_ids", row)
            self.assertNotIn("url", row)

        # schedule once: second run is a no-op (idempotent)
        r = self._cli("schedule", "once", "--topic", "local-llm-mac-mini", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        first = json.loads(r.stdout.strip())
        r = self._cli("schedule", "once", "--topic", "local-llm-mac-mini", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        second = json.loads(r.stdout.strip())
        # All objectives in topic: every second run produces zero new sources.
        for row in second:
            self.assertEqual(row["new_sources"], [],
                             f"schedule once must be idempotent, got {row}")
            self.assertIsNone(row["new_thinking"])


# ---------------------------------------------------------------------------
# E2E-6: cron triple-fire idempotency
# ---------------------------------------------------------------------------


class CronIdempotencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "cron.db")
        conn = storage.connect(self.db)
        try:
            tid = storage.create_topic(conn, "cron-topic")
            storage.create_objective(conn, tid, "best local llm mac mini")
            storage.create_objective(conn, tid, "agentic memory architecture")
        finally:
            conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _count(self) -> tuple[int, int]:
        conn = storage.connect(self.db)
        try:
            total_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            total_thinkings = conn.execute("SELECT COUNT(*) FROM thinkings").fetchone()[0]
            return int(total_sources), int(total_thinkings)
        finally:
            conn.close()

    def test_triple_fire_idempotent(self) -> None:
        schedule.run_once_for_topic(self.db, "cron-topic")
        s1, t1 = self._count()
        self.assertGreater(s1, 0)
        self.assertGreater(t1, 0)

        schedule.run_once_for_topic(self.db, "cron-topic")
        s2, t2 = self._count()
        self.assertEqual((s1, t1), (s2, t2),
                         f"second run must be a no-op; got ({s2},{t2}) vs ({s1},{t1})")

        schedule.run_once_for_topic(self.db, "cron-topic")
        s3, t3 = self._count()
        self.assertEqual((s1, t1), (s3, t3))

        # Supersession chain did not grow either.
        conn = storage.connect(self.db)
        try:
            chain_count = conn.execute(
                "SELECT COUNT(*) FROM thinkings WHERE supersedes_id IS NOT NULL"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(chain_count, 0,
                         f"cron double-fire must not create supersession rows, got {chain_count}")


# ---------------------------------------------------------------------------
# E2E-7: register_backend public-extension contract
# ---------------------------------------------------------------------------


class RegisterBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "reg.db"

    def tearDown(self) -> None:
        os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)
        if self._prev is not None:
            os.environ["OPENCLAW_RESEARCH_SEARCH"] = self._prev
        _BUILTIN_BACKENDS.pop("e2e-fake-backend", None)
        self.tmp.cleanup()

    def test_registered_backend_is_used_by_orchestrator(self) -> None:
        hits = [
            SearchHit("https://ex/reg1", "reg1", "registered backend answer one mac mini"),
            SearchHit("https://ex/reg2", "reg2", "registered backend answer two mac mini"),
        ]

        class _Fake:
            def search(self, query: str) -> list[SearchHit]:
                return list(hits)

        register_backend("e2e-fake-backend", _Fake)
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "e2e-fake-backend"

        conn = storage.connect(self.db)
        try:
            tid = storage.create_topic(conn, "reg")
            oid = storage.create_objective(conn, tid, "registered backend question")
        finally:
            conn.close()

        result = api.research(self.db, oid)
        self.assertEqual(result["mode"], "cold_start")
        self.assertEqual(len(result["new_source_ids"]), 2)
        self.assertIsNotNone(result["new_thinking_id"])


if __name__ == "__main__":
    unittest.main()
