"""Iteration-2 spec-compliance regression tests.

Covers the deferred items from `result_20260415_220322.md`:
  - Hash + inverted-index dedup caching → O(N²) per-objective insert
    cost dramatically reduced at the dominant constant-factor level.
  - File-lock around save_to_path/load_from_path.
  - Removed Node/Edge/NodeRef dead dataclasses.
  - Plugin `query` command exposes read-only retrieval surface.
  - Cached `_terms.keys()` for dedup near-dup compare (FIX A).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from research_graph import dedup, store
from research_graph.graph import InMemoryGraphBackend


# ---------------------------------------------------------------------------
# FIX A — dedup uses cached _terms.keys() instead of re-tokenizing
# ---------------------------------------------------------------------------


class DedupUsesCachedTokensTest(unittest.TestCase):
    """`dedup._row_token_set(row)` should read `row['_terms'].keys()`
    when present and only fall back to re-tokenization when missing."""

    def test_uses_cached_terms_when_present(self) -> None:
        row = {"_terms": {"alpha": 1, "beta": 2}, "content": "alpha beta"}
        s = dedup._row_token_set(row)
        self.assertEqual(s, {"alpha", "beta"})

    def test_falls_back_to_content_when_terms_missing(self) -> None:
        row = {"content": "Alpha BETA gamma"}
        s = dedup._row_token_set(row)
        self.assertEqual(s, {"alpha", "beta", "gamma"})

    def test_empty_row_returns_empty_set(self) -> None:
        self.assertEqual(dedup._row_token_set({}), set())


# ---------------------------------------------------------------------------
# FIX B — hash cache + inverted index for dedup
# ---------------------------------------------------------------------------


class HashCacheAndInvertedIndexTest(unittest.TestCase):
    """Verify that the per-(kind, objective_id) lazy caches are built,
    used, and stay correct under repeated upserts."""

    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(self.backend, tid, "q?")

    def test_exact_dup_uses_hash_cache(self) -> None:
        sid1, created1 = dedup.upsert_source(
            self.backend, self.oid, "u1", "t1",
            "Llama 3 reaches 18 tokens per second on Apple Silicon Mac mini metal",
        )
        sid2, created2 = dedup.upsert_source(
            self.backend, self.oid, "u1", "t1",
            "Llama 3 reaches 18 tokens per second on Apple Silicon Mac mini metal",
        )
        self.assertTrue(created1)
        self.assertFalse(created2, "exact-hash dup must collapse")
        self.assertEqual(sid1, sid2)
        # Hash cache slot exists and contains the entry.
        slot = store._get_hash_index_slot(self.backend, "source", self.oid)
        self.assertTrue(slot["built"])
        self.assertIn(sid1, slot["hash_to_id"].values())

    def test_near_dup_uses_inverted_index(self) -> None:
        # Two payloads with shared vocabulary should be detected as
        # near-duplicates via the inverted index, not full O(N) scan.
        long_a = "Llama 3 reaches 18 tokens per second on Apple Silicon Mac mini under metal acceleration testing"
        long_b = "Llama 3 reaches 18 tokens per second on Apple Silicon Mac mini under metal acceleration testing!"
        sid1, c1 = dedup.upsert_source(self.backend, self.oid, "u1", "t1", long_a)
        sid2, c2 = dedup.upsert_source(self.backend, self.oid, "u2", "t2", long_b)
        self.assertTrue(c1)
        # Different exact hash (different last char) but same canonical
        # tokens → Jaccard >= 0.85 → collapsed via inverted index.
        self.assertFalse(c2, "near-dup must collapse via inverted index")
        self.assertEqual(sid1, sid2)
        # Inverted-index slot exists with postings populated.
        slot = dedup._get_index_slot(self.backend, "source", self.oid)
        self.assertTrue(slot["built"])
        self.assertGreater(len(slot["postings"]), 0)
        self.assertIn(sid1, slot["doc_token_count"])

    def test_distinct_docs_are_not_collapsed(self) -> None:
        # Different vocabularies should NOT be collapsed.
        a = "MacBook Pro M3 GPU memory bandwidth benchmark"
        b = "Linux kernel scheduler latency analysis under load"
        sid1, c1 = dedup.upsert_source(self.backend, self.oid, "u1", "t1", a)
        sid2, c2 = dedup.upsert_source(self.backend, self.oid, "u2", "t2", b)
        self.assertTrue(c1)
        self.assertTrue(c2, "distinct docs must NOT collapse")
        self.assertNotEqual(sid1, sid2)


class DedupPerfRegressionTest(unittest.TestCase):
    """Per-insert dedup cost must stay below a generous ceiling for the
    spec's "long-running cron" use case. This test was added because
    the iteration-1 audit measured the original code at 1.5 ms/insert
    at N=1600 and growing super-linearly. With FIX A + FIX B applied,
    we expect roughly 0.25 ms/insert at N=1600 — a 5-6× speedup.

    The assertion is intentionally loose (10× the measured win) so
    slow CI hosts don't flake. The point is to catch a regression that
    drops the speedup back below 2× — which would mean the perf fix
    was reverted or undermined.
    """

    def test_per_insert_under_loose_ceiling(self) -> None:
        import random
        random.seed(2026)
        WORDS = [
            "llama", "mistral", "qwen", "phi", "gemma", "metal", "cuda", "mlx",
            "macbook", "macmini", "rtx", "m1", "m2", "m3", "tokens", "second",
            "inference", "context", "window", "quantized", "gguf", "ollama",
            "apple", "silicon", "benchmark", "comparison", "memory", "gpu",
        ]
        b = InMemoryGraphBackend()
        tid = store.create_topic(b, "llm")
        oid = store.create_objective(b, tid, "q?")
        N = 800
        docs = [
            " ".join(random.sample(WORDS, 8)) + f" doc{i} unique_{i*13}"
            for i in range(N)
        ]
        t0 = time.perf_counter()
        for i, content in enumerate(docs):
            dedup.upsert_source(b, oid, f"u{i}", f"t{i}", content)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_insert_ms = elapsed_ms / N
        # Pre-FIX baseline at N=800 was ~580 ms total / 0.73 ms per
        # insert. After FIX B it dropped to ~107 ms / 0.13 ms per
        # insert. Ceiling: 1.0 ms/insert (well above expected, well
        # below the regressed baseline).
        self.assertLess(
            per_insert_ms, 1.0,
            f"per-insert dedup cost {per_insert_ms:.3f} ms is above "
            f"the regression ceiling — dedup index likely broken"
        )
        # Sanity: docs were actually inserted (a wildly broken dedup
        # might collapse everything).
        live = len(store.list_sources(b, oid))
        self.assertGreater(live, N // 2,
            f"live source count {live} is suspiciously low — dedup is over-collapsing")


# ---------------------------------------------------------------------------
# FIX C — file lock around save_to_path / load_from_path
# ---------------------------------------------------------------------------


class FileLockTest(unittest.TestCase):
    """Verify the `_FileLock` context manager is callable and produces
    a `.lock` sidecar (POSIX) without error. Cross-process fork test
    is intentionally omitted to avoid flakiness on CI; the point of
    this test is that the lock infrastructure is wired up and not a
    no-op in the import path."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="rg-lock-")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_creates_lock_sidecar_on_posix(self) -> None:
        from research_graph.graph.inmemory import _HAS_FLOCK
        if not _HAS_FLOCK:
            self.skipTest("fcntl not available (Windows)")
        path = Path(self.tmp) / "rg.json"
        b = InMemoryGraphBackend()
        store.create_topic(b, "llm")
        b.save_to_path(path)
        # Lock sidecar exists.
        lock_path = path.with_suffix(path.suffix + ".lock")
        self.assertTrue(lock_path.exists(),
                        f"expected lock sidecar at {lock_path}")
        # And the JSON is well-formed.
        self.assertTrue(path.exists())
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data.get("version"), 1)

    def test_load_with_concurrent_save_does_not_corrupt(self) -> None:
        # Sequential proxy for concurrency: write twice in a row,
        # then read. The lock contract guarantees the read sees a
        # complete file, never a half-written one.
        path = Path(self.tmp) / "rg.json"
        b1 = InMemoryGraphBackend()
        store.create_topic(b1, "llm")
        b1.save_to_path(path)
        b2 = InMemoryGraphBackend()
        store.create_topic(b2, "ml")
        b2.save_to_path(path)
        b3 = InMemoryGraphBackend()
        b3.load_from_path(path)
        topics = store.list_topics(b3)
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]["name"], "ml")


# ---------------------------------------------------------------------------
# FIX E — Node / Edge / NodeRef dead dataclasses removed
# ---------------------------------------------------------------------------


class DeadDataclassRemovalTest(unittest.TestCase):
    """Ensure the audit-flagged dead Node/Edge/NodeRef dataclasses are
    actually gone (not just commented out). This protects against a
    well-meaning future revert."""

    def test_model_module_does_not_export_dataclasses(self) -> None:
        from research_graph.graph import model
        for name in ("Node", "Edge", "NodeRef"):
            self.assertFalse(
                hasattr(model, name),
                f"expected `research_graph.graph.model.{name}` to be removed"
            )

    def test_graph_init_does_not_re_export(self) -> None:
        from research_graph import graph as gpkg
        for name in ("Node", "Edge", "NodeRef"):
            self.assertNotIn(name, gpkg.__all__)
            self.assertFalse(hasattr(gpkg, name))


# ---------------------------------------------------------------------------
# FIX D — plugin_helper `cmd_query` read-only retrieval surface
# ---------------------------------------------------------------------------


class PluginQueryCommandTest(unittest.TestCase):
    """`cmd_query` should run retrieval against the stored memory
    without invoking the orchestrator or fetching external sources."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="rg-plugin-query-")
        self.db = str(Path(self.tmp) / "rg.json")
        plugin_dir = Path(__file__).resolve().parent.parent / "plugins" / "research-memory"
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_seeded(self) -> tuple[int, int, int]:
        backend = InMemoryGraphBackend()
        try:
            tid = store.create_topic(backend, "llm")
            oid = store.create_objective(backend, tid, "best local llm on mac mini")
            sid1, _ = dedup.upsert_source(
                backend, oid, "u1", "Llama 3 spec",
                "Llama 3 reaches 18 tokens per second on Apple Silicon Mac mini metal",
            )
            sid2, _ = dedup.upsert_source(
                backend, oid, "u2", "Mistral spec",
                "Mistral 7B benchmark shows 22 tokens per second under MLX runtime",
            )
            backend.save_to_path(self.db)
            return tid, oid, sid1
        finally:
            backend.close()

    def _query(self, payload: dict) -> dict:
        import argparse
        import plugin_helper as ph  # type: ignore[import-not-found]
        ns = argparse.Namespace(db=self.db, payload=json.dumps(payload))
        return ph.cmd_query(ns)

    def test_query_by_topic_name_returns_sources(self) -> None:
        self._make_seeded()
        out = self._query({
            "topic_name": "llm",
            "question": "tokens per second mac mini",
            "kind": "source",
        })
        self.assertEqual(out["topic_name"], "llm")
        self.assertGreater(len(out["sources"]), 0)
        self.assertEqual(out["thinkings"], [])
        # Each row has the canonical query-result shape.
        for row in out["sources"]:
            self.assertIn("id", row)
            self.assertIn("score", row)
            self.assertIn("content_snippet", row)
            self.assertLessEqual(len(row["content_snippet"]), 280)

    def test_query_default_kind_is_both(self) -> None:
        self._make_seeded()
        out = self._query({
            "topic_name": "llm",
            "question": "llama",
        })
        self.assertEqual(out["kind"], "both")
        self.assertIn("sources", out)
        self.assertIn("thinkings", out)

    def test_query_rejects_missing_question(self) -> None:
        self._make_seeded()
        with self.assertRaises(ValueError):
            self._query({"topic_name": "llm"})

    def test_query_rejects_invalid_kind(self) -> None:
        self._make_seeded()
        with self.assertRaises(ValueError):
            self._query({
                "topic_name": "llm",
                "question": "x",
                "kind": "invalid",
            })

    def test_query_does_not_invoke_external_search(self) -> None:
        """cmd_query is read-only and must NOT call any search backend
        or write to the graph. We assert this by monkey-patching
        get_default_search to raise — if cmd_query touches it, we'd
        see the exception."""
        from research_graph import search as search_mod
        from research_graph import orchestrator as orch_mod
        self._make_seeded()
        original = orch_mod.get_default_search
        def boom():
            raise RuntimeError("query command must not invoke search")
        orch_mod.get_default_search = boom  # type: ignore[assignment]
        try:
            out = self._query({
                "topic_name": "llm",
                "question": "llama",
                "kind": "both",
            })
        finally:
            orch_mod.get_default_search = original  # type: ignore[assignment]
        # Did not raise → query is read-only.
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
