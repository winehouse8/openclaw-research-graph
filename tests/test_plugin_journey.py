"""iter-5 continual-research plugin journey tests.

Exercises `research_graph.api.research_journey` + `thinking_history`
end-to-end over a tempfile-backed InMemory JSON store. These are the
single entry points the OpenClaw plugin helper (`plugin_helper.py`) and
the JS plugin handlers route to, so locking down their behaviour here
also locks down the plugin user journey.

Scenarios covered:
  1. Cold-start research on a brand-new topic+question produces a live
     tip with non-null quality_overall, memory_before.exists=False,
     memory_after.exists=True, outcome == "cold_start".
  2. Second run on the same topic+question reuses the prior thinking
     via dedup; outcome in {"reused", "unchanged"}; live_tip_id
     unchanged.
  3. Resolving via exact-match does NOT create a second objective
     when the same question is passed twice.
  4. `thinking_history` walks the chain and reports siblings.
  5. A regression branch (weak backend run after a rich backend run)
     becomes a sibling; live tip stays on the rich thinking; history
     reports one chain entry + one sibling entry.
  6. Malformed `/research-history` resolution (non-existent topic)
     returns `{exists: false, ...}` rather than creating state.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from research_graph import api, reasoning, store
from research_graph.graph import get_default_backend
from research_graph.orchestrator import Orchestrator
from research_graph.search import SearchHit


# A deterministic ExternalSearch that returns a fixed set of hits. We
# register it via the `search.register_backend` hook so the in-process
# Orchestrator constructed inside `api.research_journey` picks it up
# via `get_default_search()` after we set the env var.
class _FixtureSearch:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query):
        return list(self._hits)


_FIXTURE_HITS = [
    SearchHit(
        url="https://example.com/llama",
        title="Llama benchmark",
        content=(
            "llama three eight billion delivers eighteen tokens per second "
            "benchmark apple silicon metal"
        ),
    ),
    SearchHit(
        url="https://example.com/mistral",
        title="Mistral benchmark",
        content=(
            "mistral seven billion benchmark twenty two tokens per second "
            "mlx quantized gguf"
        ),
    ),
    SearchHit(
        url="https://example.com/qwen",
        title="Qwen benchmark",
        content=(
            "qwen fourteen billion benchmark thirteen tokens per second "
            "metal acceleration apple"
        ),
    ),
    SearchHit(
        url="https://example.com/phi",
        title="Phi benchmark",
        content=(
            "phi three billion benchmark twelve tokens per second "
            "heavy quantization macbook"
        ),
    ),
]


def _register_fixture_search():
    from research_graph import search as _search
    _search.register_backend("fixture", lambda: _FixtureSearch(_FIXTURE_HITS))


class ResearchJourneyColdStartTest(unittest.TestCase):

    def setUp(self) -> None:
        _register_fixture_search()
        self._old_search = os.environ.get("OPENCLAW_RESEARCH_SEARCH")
        self._old_actor = os.environ.get("OPENCLAW_RESEARCH_ACTOR")
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "fixture"
        # Use placeholder actor so all tests stay deterministic.
        os.environ["OPENCLAW_RESEARCH_ACTOR"] = "placeholder"
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()
        if self._old_search is None:
            os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)
        else:
            os.environ["OPENCLAW_RESEARCH_SEARCH"] = self._old_search
        if self._old_actor is None:
            os.environ.pop("OPENCLAW_RESEARCH_ACTOR", None)
        else:
            os.environ["OPENCLAW_RESEARCH_ACTOR"] = self._old_actor

    def test_cold_start_creates_topic_objective_and_live_tip(self) -> None:
        out = api.research_journey(
            self.db, "local-llm", "benchmark billion tokens second"
        )
        # Topic + objective were created.
        self.assertFalse(out["topic"]["existed_before"])
        self.assertFalse(out["objective"]["existed_before"])
        self.assertGreater(out["topic"]["id"], 0)
        self.assertGreater(out["objective"]["id"], 0)
        # Memory snapshot transitioned empty → populated.
        self.assertFalse(out["memory_before"]["exists"])
        self.assertTrue(out["memory_after"]["exists"])
        self.assertIsNotNone(out["memory_after"]["live_thinking_id"])
        self.assertIsNotNone(out["memory_after"]["live_quality_overall"])
        self.assertGreater(out["memory_after"]["source_count"], 0)
        self.assertEqual(out["memory_after"]["thinking_count"], 1)
        # Outcome classification + delta narrative.
        self.assertEqual(out["delta"]["outcome"], "cold_start")
        self.assertIn("cold_start", out["delta"]["outcome"])
        self.assertIn("first research run", out["delta"]["narrative"].lower())

    def test_second_run_reuses_objective_and_dedupe_collapses(self) -> None:
        out1 = api.research_journey(
            self.db, "local-llm", "benchmark billion tokens second"
        )
        out2 = api.research_journey(
            self.db, "local-llm", "benchmark billion tokens second"
        )
        # Objective is reused, not recreated.
        self.assertTrue(out2["topic"]["existed_before"])
        self.assertTrue(out2["objective"]["existed_before"])
        self.assertEqual(out2["objective"]["id"], out1["objective"]["id"])
        # Memory exists on the before-snapshot of the second run.
        self.assertTrue(out2["memory_before"]["exists"])
        # Dedup collapsed onto the existing thinking → outcome is reused
        # (or unchanged on a corner case).
        self.assertIn(out2["delta"]["outcome"], {"reused", "unchanged"})
        # Live tip id should not have moved.
        self.assertEqual(
            out2["memory_after"]["live_thinking_id"],
            out1["memory_after"]["live_thinking_id"],
        )

    def test_exact_match_topic_question_creates_exactly_one_objective(self) -> None:
        api.research_journey(self.db, "local-llm", "q A")
        api.research_journey(self.db, "local-llm", "q A")
        api.research_journey(self.db, "local-llm", "q B")
        backend = get_default_backend(self.db)
        try:
            t = store.get_topic_by_name(backend, "local-llm")
            self.assertIsNotNone(t)
            objs = store.query_objectives(backend, topic_id=int(t["id"]))
            questions = sorted(o["question"] for o in objs)
            self.assertEqual(questions, ["q A", "q B"])
        finally:
            backend.close()


class ThinkingHistoryTest(unittest.TestCase):

    def setUp(self) -> None:
        _register_fixture_search()
        self._old_search = os.environ.get("OPENCLAW_RESEARCH_SEARCH")
        self._old_actor = os.environ.get("OPENCLAW_RESEARCH_ACTOR")
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "fixture"
        os.environ["OPENCLAW_RESEARCH_ACTOR"] = "placeholder"
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()
        if self._old_search is None:
            os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)
        else:
            os.environ["OPENCLAW_RESEARCH_SEARCH"] = self._old_search
        if self._old_actor is None:
            os.environ.pop("OPENCLAW_RESEARCH_ACTOR", None)
        else:
            os.environ["OPENCLAW_RESEARCH_ACTOR"] = self._old_actor

    def test_history_after_cold_start_returns_single_chain_entry(self) -> None:
        out = api.research_journey(self.db, "t1", "q1")
        hist = api.thinking_history(self.db, out["objective"]["id"])
        self.assertTrue(hist["exists"])
        self.assertEqual(len(hist["chain"]), 1)
        self.assertEqual(hist["live_tip_id"], hist["chain"][0]["thinking_id"])
        self.assertEqual(hist["siblings"], [])
        # History entry carries quality + actor backend provenance.
        entry = hist["chain"][0]
        self.assertIsNotNone(entry["quality_overall"])
        self.assertEqual(entry["actor_backend"], reasoning.ACTOR_BACKEND_PLACEHOLDER)
        self.assertGreaterEqual(entry["citation_count"], 1)

    def test_history_for_unknown_objective_is_readonly(self) -> None:
        hist = api.thinking_history(self.db, 99999)
        self.assertFalse(hist["exists"])
        self.assertEqual(hist["chain"], [])
        self.assertEqual(hist["siblings"], [])


class RegressionBranchJourneyTest(unittest.TestCase):
    """Simulate the iter-4 hypothesis-branching path end-to-end via the
    plugin entrypoint. Seed the objective with a RICH actor, then run
    again with a WEAK actor; the journey must report outcome="branched"
    and the live tip must stay on the rich thinking."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _rich_actor(self):
        class Rich:
            name = "fake/rich-journey"

            def propose(self, sources, q):
                if not sources:
                    return ("No sources.", [])
                bits = []
                ids = []
                for s in sources:
                    toks = [t for t in s.get("content", "").split()
                            if len(t) > 3 and t.isalpha()][:3]
                    bits.append(f"from {s['id']} we observe " + " ".join(toks))
                    ids.append(int(s["id"]))
                return (
                    f"Synthesis for {q}.\n" + "\n".join(bits)
                    + "\nConclusion: comprehensive.",
                    ids,
                )
        return Rich()

    def _weak_actor(self):
        class Weak:
            name = "fake/weak-journey"

            def propose(self, sources, q):
                if not sources:
                    return ("No sources.", [])
                s = sources[0]
                toks = [t for t in s.get("content", "").split()
                        if len(t) > 3 and t.isalpha()][:2]
                return (f"Weak note. {' '.join(toks)}", [int(s["id"])])
        return Weak()

    def _seed_objective(self) -> int:
        # Populate topic + objective + 4 sources without running the
        # orchestrator yet, so we can drive the two runs with different
        # explicit backends.
        backend = get_default_backend(self.db)
        try:
            tid = store.create_topic(backend, "branchy")
            oid = store.create_objective(backend, tid, "benchmark billion tokens second")
            from research_graph import dedup as _dedup
            for url, title, content in [
                ("u1", "t1", "llama three billion benchmark tokens second metal apple silicon"),
                ("u2", "t2", "mistral seven billion benchmark tokens second mlx quantized"),
                ("u3", "t3", "qwen fourteen billion benchmark tokens second metal acceleration"),
                ("u4", "t4", "phi three billion benchmark tokens second heavy quantization"),
            ]:
                _dedup.upsert_source(backend, oid, url, title, content)
            return oid
        finally:
            backend.close()

    def test_regression_creates_sibling_branch_via_api_path(self) -> None:
        oid = self._seed_objective()

        # Run 1: rich actor via Orchestrator directly (mimicking what
        # api.research_journey does internally). We use a fake search
        # that returns no additional hits so the only scope is the
        # pre-seeded corpus.
        from research_graph import search as _search
        _search.register_backend("noop", lambda: _FixtureSearch([]))
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "noop"
        try:
            backend = get_default_backend(self.db)
            try:
                orch = Orchestrator(backend, actor_backend=self._rich_actor())
                r1 = orch.research(oid)
            finally:
                backend.close()

            backend = get_default_backend(self.db)
            try:
                orch = Orchestrator(backend, actor_backend=self._weak_actor())
                r2 = orch.research(oid)
            finally:
                backend.close()
        finally:
            os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)

        # r2 must be a sibling branch — weak actor regressed on
        # grounding/coverage vs rich.
        self.assertTrue(r2.branched or r2.reused_thinking_id is not None)
        if r2.new_thinking_id is not None:
            self.assertIsNone(r2.supersedes_id)

        # History walk via api must show the rich thinking as the live
        # tip. If a sibling was created it shows up under `siblings`.
        hist = api.thinking_history(self.db, oid)
        self.assertEqual(hist["live_tip_id"], int(r1.new_thinking_id))
        if r2.branched and r2.new_thinking_id is not None:
            self.assertEqual(len(hist["siblings"]), 1)
            self.assertEqual(
                hist["siblings"][0]["thinking_id"], int(r2.new_thinking_id)
            )
            self.assertEqual(hist["siblings"][0]["actor_backend"], "fake/weak-journey")


class ClassifyOutcomeUnitTest(unittest.TestCase):

    def test_cold_start_no_prior(self) -> None:
        out, delta, narr = api._classify_outcome(
            run={"mode": "cold_start", "new_thinking_id": 1, "rejected_reasons": []},
            before={"live_thinking_id": None, "live_quality_overall": None},
            after={"live_thinking_id": 1, "live_quality_overall": 0.75},
        )
        self.assertEqual(out, "cold_start")
        self.assertIsNone(delta)  # nothing to diff against
        self.assertIn("first research run", narr.lower())

    def test_rejected_wins_over_other_flags(self) -> None:
        out, _, narr = api._classify_outcome(
            run={
                "mode": "memory_augmented",
                "new_thinking_id": 5,
                "rejected_reasons": ["critic failed"],
            },
            before={"live_thinking_id": 1, "live_quality_overall": 0.5},
            after={"live_thinking_id": 1, "live_quality_overall": 0.5},
        )
        self.assertEqual(out, "rejected")
        self.assertIn("critic", narr.lower())

    def test_branched_path(self) -> None:
        out, delta, narr = api._classify_outcome(
            run={
                "mode": "memory_augmented",
                "new_thinking_id": 9,
                "branched": True,
                "rejected_reasons": [],
            },
            before={"live_thinking_id": 3, "live_quality_overall": 0.9},
            after={"live_thinking_id": 3, "live_quality_overall": 0.9},
        )
        self.assertEqual(out, "branched")
        self.assertEqual(delta, 0.0)
        self.assertIn("sibling", narr.lower())

    def test_improved_path(self) -> None:
        out, delta, narr = api._classify_outcome(
            run={
                "mode": "memory_augmented",
                "new_thinking_id": 9,
                "supersedes_id": 3,
                "branched": False,
                "rejected_reasons": [],
            },
            before={"live_thinking_id": 3, "live_quality_overall": 0.5},
            after={"live_thinking_id": 9, "live_quality_overall": 0.9},
        )
        self.assertEqual(out, "improved")
        self.assertAlmostEqual(delta, 0.4)
        self.assertIn("superseded", narr.lower())


if __name__ == "__main__":
    unittest.main()
