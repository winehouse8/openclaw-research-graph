"""iter-4 continual-research regression tests.

Locks in the three structural changes that make the system actually
scale with reasoning horsepower:

  1. Pluggable `reasoning.ActorBackend` — Orchestrator resolves via
     registry + env var, mirroring `research_graph.search`.
  2. `reasoning.score_thinking` — quantitative quality metric
     (grounding / coverage / diversity / overall).
  3. Quality-gated supersession in the Orchestrator — a new Thinking
     that regresses on overall quality becomes a sibling branch
     rather than overwriting the live tip (hypothesis branching).
"""
from __future__ import annotations

import os
import unittest

from research_graph import dedup, reasoning, retrieval, store
from research_graph.graph import InMemoryGraphBackend
from research_graph.orchestrator import (
    Orchestrator,
    QUALITY_REGRESSION_EPSILON,
    ResearchResult,
)
from research_graph.search import SearchHit


# ---------------------------------------------------------------------------
# Fake ActorBackends used by several tests below.
# ---------------------------------------------------------------------------


class _RichActorBackend:
    """Actor that emits a grounded, diverse thinking. Returns a body
    that reuses tokens from EVERY cited source, so grounding is 1.0,
    and cites every source in scope, so coverage is 1.0."""

    name = "fake/rich-v1"

    def propose(self, sources, question):
        if not sources:
            return ("No sources.", [])
        bits = []
        ids = []
        for s in sources:
            # Pick 3 meaningful tokens from each source so grounding
            # has signal on every citation.
            toks = [
                t for t in s.get("content", "").split()
                if len(t) > 3 and t.isalpha()
            ][:3]
            bits.append(f"from {s['id']} we observe " + " ".join(toks))
            ids.append(int(s["id"]))
        body = (
            f"Synthesis of {question}.\n"
            + "\n".join(bits)
            + "\nConclusion: integrated across all cited sources."
        )
        return (body, ids)


class _WeakActorBackend:
    """Actor that cites only one source and uses generic filler text.
    grounding → ~1.0 for that one citation, coverage → 1/|scope|
    which will be low on a multi-source scope, diversity → 1.0.
    Overall ends up well below rich backend on a 4+ source scope."""

    name = "fake/weak-v1"

    def propose(self, sources, question):
        if not sources:
            return ("No sources.", [])
        s = sources[0]
        toks = [
            t for t in s.get("content", "").split()
            if len(t) > 3 and t.isalpha()
        ][:2]
        body = (
            f"Synthesis of {question}. One source suffices: "
            + " ".join(toks)
        )
        return (body, [int(s["id"])])


class _DeterministicSearch:
    """Fake ExternalSearch that always returns the same multi-source
    fixture. We use this instead of the OfflineFixtureSearch so tests
    don't depend on the fixture file contents."""

    def __init__(self, hits):
        self._hits = hits

    def search(self, query):
        return list(self._hits)


_OBJECTIVE_QUESTION = "benchmark billion tokens second"


def _sources_raw():
    # Every content contains at least TWO of the query tokens so the
    # retrieval candidate filter (iter-3 postings prune) returns all
    # four documents as scope. Without this, a query whose rare
    # tokens live in only one doc would collapse the scope to that
    # one doc and coverage would trivialise to 1.0 for every actor.
    return [
        ("https://a.example/llama", "Llama post",
         "llama three eight billion delivers eighteen tokens per second benchmark apple silicon metal"),
        ("https://b.example/mistral", "Mistral post",
         "mistral seven billion benchmark shows twenty two tokens per second mlx quantized gguf"),
        ("https://c.example/qwen", "Qwen post",
         "qwen fourteen billion benchmark competitive thirteen tokens per second m series metal"),
        ("https://d.example/phi", "Phi post",
         "phi three billion maintains quality benchmark twelve tokens per second heavy quantization"),
    ]


def _make_fixture_backend_with_sources():
    """Build an in-memory backend + topic + objective with 4 distinct
    sources so coverage/diversity math has non-trivial values."""
    backend = InMemoryGraphBackend()
    tid = store.create_topic(backend, "llm")
    oid = store.create_objective(backend, tid, _OBJECTIVE_QUESTION)
    for url, title, content in _sources_raw():
        dedup.upsert_source(backend, oid, url, title, content)
    return backend, oid


# ---------------------------------------------------------------------------
# Pluggable ActorBackend registry
# ---------------------------------------------------------------------------


class ActorBackendRegistryTest(unittest.TestCase):

    def test_placeholder_is_default(self):
        ab = reasoning.get_default_actor_backend()
        self.assertEqual(ab.name, "placeholder/keyword-extractor")

    def test_env_var_dispatch(self):
        reasoning.register_actor_backend("rich-test", _RichActorBackend)
        old = os.environ.get("OPENCLAW_RESEARCH_ACTOR")
        try:
            os.environ["OPENCLAW_RESEARCH_ACTOR"] = "rich-test"
            ab = reasoning.get_default_actor_backend()
            self.assertEqual(ab.name, "fake/rich-v1")
        finally:
            if old is None:
                os.environ.pop("OPENCLAW_RESEARCH_ACTOR", None)
            else:
                os.environ["OPENCLAW_RESEARCH_ACTOR"] = old

    def test_unknown_backend_raises(self):
        old = os.environ.get("OPENCLAW_RESEARCH_ACTOR")
        try:
            os.environ["OPENCLAW_RESEARCH_ACTOR"] = "definitely-not-registered"
            with self.assertRaises(ValueError):
                reasoning.get_default_actor_backend()
        finally:
            if old is None:
                os.environ.pop("OPENCLAW_RESEARCH_ACTOR", None)
            else:
                os.environ["OPENCLAW_RESEARCH_ACTOR"] = old

    def test_legacy_actor_propose_still_works(self):
        # Pre-iter-4 free-function entrypoint must still return exactly
        # what it used to return so any test / caller pinned to the
        # keyword-extractor output keeps passing.
        sources = [
            {"id": 1, "url": "u", "title": "t", "content": "alpha beta gamma"},
        ]
        text, cited = reasoning.actor_propose(sources, "q?")
        self.assertIn("Synthesis for: q?", text)
        self.assertIn("keywords:", text)
        self.assertEqual(cited, [1])


# ---------------------------------------------------------------------------
# QualityScore math
# ---------------------------------------------------------------------------


class QualityScoreTest(unittest.TestCase):

    def test_empty_citations_scores_zero(self):
        q = reasoning.score_thinking("anything", [])
        self.assertEqual(q.overall, 0.0)
        self.assertIn("no citations", q.notes)

    def test_full_grounding_full_coverage_full_diversity(self):
        sources = [
            {"id": 1, "url": "u1", "content": "alpha beta"},
            {"id": 2, "url": "u2", "content": "gamma delta"},
        ]
        # Thinking body mentions tokens from both sources → grounding 1.0
        # Cites both sources, scope is both → coverage 1.0
        # Two distinct URLs on 2 citations → diversity 1.0
        q = reasoning.score_thinking(
            "we note alpha and gamma together", sources, scope_sources=sources
        )
        self.assertAlmostEqual(q.grounding, 1.0)
        self.assertAlmostEqual(q.coverage, 1.0)
        self.assertAlmostEqual(q.diversity, 1.0)
        self.assertAlmostEqual(q.overall, 1.0)

    def test_partial_grounding(self):
        sources = [
            {"id": 1, "url": "u1", "content": "alpha beta"},
            {"id": 2, "url": "u2", "content": "gamma delta"},
        ]
        # Thinking mentions source 1 tokens but NOT source 2 tokens →
        # grounding = 1/2 = 0.5
        q = reasoning.score_thinking(
            "alpha and beta are relevant", sources, scope_sources=sources
        )
        self.assertAlmostEqual(q.grounding, 0.5)

    def test_coverage_drops_when_scope_bigger_than_citations(self):
        cited = [{"id": 1, "url": "u1", "content": "alpha beta"}]
        scope = cited + [
            {"id": 2, "url": "u2", "content": "gamma"},
            {"id": 3, "url": "u3", "content": "delta"},
            {"id": 4, "url": "u4", "content": "epsilon"},
        ]
        q = reasoning.score_thinking("alpha beta", cited, scope_sources=scope)
        self.assertAlmostEqual(q.coverage, 0.25)  # 1/4

    def test_diversity_drops_on_repeated_url(self):
        sources = [
            {"id": 1, "url": "u1", "content": "alpha"},
            {"id": 2, "url": "u1", "content": "alpha"},  # same URL
            {"id": 3, "url": "u1", "content": "alpha"},
        ]
        q = reasoning.score_thinking("alpha", sources)
        self.assertAlmostEqual(q.diversity, 1 / 3)

    def test_to_dict_is_json_safe(self):
        q = reasoning.score_thinking(
            "alpha", [{"id": 1, "url": "u1", "content": "alpha"}]
        )
        d = q.to_dict()
        import json
        json.dumps(d)  # must not raise
        self.assertEqual(
            set(d.keys()),
            {"grounding", "coverage", "diversity", "overall",
             "cited_count", "scope_count", "notes"},
        )


# ---------------------------------------------------------------------------
# Orchestrator end-to-end with pluggable backend + quality persistence
# ---------------------------------------------------------------------------


class OrchestratorQualityIntegrationTest(unittest.TestCase):

    def _make_orch(self, backend, actor_backend_cls):
        hits = [
            SearchHit(url=u, title=t, content=c) for u, t, c in _sources_raw()
        ]
        return Orchestrator(
            backend,
            search=_DeterministicSearch(hits),
            actor_backend=actor_backend_cls(),
        )

    def test_rich_backend_produces_high_quality_thinking(self):
        backend, oid = _make_fixture_backend_with_sources()
        orch = self._make_orch(backend, _RichActorBackend)
        result = orch.research(oid)
        self.assertIsNotNone(result.new_thinking_id)
        self.assertEqual(result.actor_backend, "fake/rich-v1")
        self.assertIsNotNone(result.quality_score)
        self.assertGreaterEqual(result.quality_score["overall"], 0.6)
        # Row persists quality metadata.
        row = store.get_thinking(backend, int(result.new_thinking_id))
        self.assertIsNotNone(row)
        self.assertEqual(row["actor_backend"], "fake/rich-v1")
        self.assertIsNotNone(row.get("quality_score"))
        self.assertIsNotNone(row.get("quality_overall"))

    def test_weak_backend_produces_lower_quality_than_rich(self):
        # Same objective seeded fresh twice — we compare the FIRST
        # thinking each backend would produce (not a day-2 supersession
        # scenario, just isolated quality comparison).
        backend_a, oid_a = _make_fixture_backend_with_sources()
        backend_b, oid_b = _make_fixture_backend_with_sources()
        rich = self._make_orch(backend_a, _RichActorBackend).research(oid_a)
        weak = self._make_orch(backend_b, _WeakActorBackend).research(oid_b)
        self.assertGreater(
            rich.quality_score["overall"],
            weak.quality_score["overall"],
            "rich backend must outscore weak backend on the same scope",
        )


# ---------------------------------------------------------------------------
# Quality-gated supersession / hypothesis branching
# ---------------------------------------------------------------------------


class QualityGatedSupersessionTest(unittest.TestCase):
    """The headline continual-research change: day-2 cannot overwrite
    day-1 with a worse answer."""

    def _hits_day1(self):
        return [
            SearchHit(url=u, title=t, content=c) for u, t, c in _sources_raw()
        ]

    def _hits_day2(self):
        # Day-2 adds a new source so the orchestrator sees fresh
        # evidence (the pre-iter-4 supersession policy would always
        # overwrite here regardless of quality). The day-2 hit also
        # shares multiple query tokens so retrieval picks it up.
        return self._hits_day1() + [
            SearchHit(
                url="https://e.example/day2", title="Day 2 post",
                content="gemma four billion benchmark shows fifteen tokens per second new apple silicon metal",
            ),
        ]

    def test_regression_blocked_new_thinking_becomes_sibling(self):
        backend = InMemoryGraphBackend()
        tid = store.create_topic(backend, "llm")
        oid = store.create_objective(backend, tid, _OBJECTIVE_QUESTION)

        day1_orch = Orchestrator(
            backend,
            search=_DeterministicSearch(self._hits_day1()),
            actor_backend=_RichActorBackend(),
        )
        day1 = day1_orch.research(oid)
        self.assertIsNotNone(day1.new_thinking_id)
        day1_overall = day1.quality_score["overall"]
        day1_thinking_id = int(day1.new_thinking_id)

        day2_orch = Orchestrator(
            backend,
            search=_DeterministicSearch(self._hits_day2()),
            actor_backend=_WeakActorBackend(),
        )
        day2 = day2_orch.research(oid)

        # Day-2 must be worse than day-1 (preamble for this test).
        self.assertLess(
            day2.quality_score["overall"],
            day1_overall - QUALITY_REGRESSION_EPSILON,
            "weak backend should produce a clear regression vs rich backend",
        )

        # If a new thinking row got inserted, it must be a SIBLING, not
        # a supersession. If dedup collapsed onto day-1 that's also
        # fine — the live tip must remain day-1 either way.
        if day2.new_thinking_id is not None:
            self.assertTrue(day2.branched, "weak new thinking should be branched")
            self.assertIsNone(day2.supersedes_id)
            # The new thinking must NOT be the live tip.
            live = store.latest_live_thinking(backend, oid)
            self.assertIsNotNone(live)
            self.assertEqual(int(live["id"]), day1_thinking_id,
                             "live tip must remain the rich day-1 thinking")

    def test_improvement_allowed_supersedes(self):
        backend = InMemoryGraphBackend()
        tid = store.create_topic(backend, "llm")
        oid = store.create_objective(backend, tid, _OBJECTIVE_QUESTION)

        # Day-1 with WEAK backend.
        day1 = Orchestrator(
            backend,
            search=_DeterministicSearch(self._hits_day1()),
            actor_backend=_WeakActorBackend(),
        ).research(oid)
        self.assertIsNotNone(day1.new_thinking_id)
        day1_overall = day1.quality_score["overall"]

        # Day-2 with RICH backend + new evidence.
        day2 = Orchestrator(
            backend,
            search=_DeterministicSearch(self._hits_day2()),
            actor_backend=_RichActorBackend(),
        ).research(oid)
        self.assertIsNotNone(day2.new_thinking_id)
        self.assertGreater(day2.quality_score["overall"], day1_overall)
        self.assertFalse(day2.branched)
        self.assertEqual(day2.supersedes_id, int(day1.new_thinking_id))
        live = store.latest_live_thinking(backend, oid)
        self.assertIsNotNone(live)
        self.assertEqual(int(live["id"]), int(day2.new_thinking_id))


if __name__ == "__main__":
    unittest.main()
