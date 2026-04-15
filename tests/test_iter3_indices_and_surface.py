"""Iteration-3 spec-compliance regression tests.

Closes the deferred items from `result_20260416_013504.md`:
  - `inmemory.has_edge` O(N) → O(1) via sidecar `_out_keys` set
  - Global retrieval index (DF + postings) for sub-linear
    candidate filtering inside `retrieval._rank`
  - Plugin `update` / `delete` commands for CRUQD U/D parity at
    the OpenClaw integration boundary
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from research_graph import dedup, retrieval, store
from research_graph.graph import InMemoryGraphBackend


# ---------------------------------------------------------------------------
# inmemory.has_edge → O(1) via sidecar set
# ---------------------------------------------------------------------------


class HasEdgeO1Test(unittest.TestCase):
    """has_edge previously scanned `self._out[src]` linearly. Now it
    uses the `_out_keys[src]` set populated in lockstep by
    `create_edge` / `_delete_single` / `load_from_path`. We test:
      - existing edge → True
      - non-existing edge → False
      - duplicate create_edge does NOT double-insert
      - delete drops the edge from the set
      - JSON round-trip preserves the set"""

    def setUp(self) -> None:
        self.b = InMemoryGraphBackend()
        self.t = self.b.create_node("Topic", {"name": "t"})
        self.o = self.b.create_node("Objective", {"q": "?"})

    def test_has_edge_basic(self) -> None:
        self.b.create_edge(self.t, "HAS_OBJECTIVE", self.o)
        self.assertTrue(self.b.has_edge(self.t, "HAS_OBJECTIVE", self.o))
        self.assertFalse(self.b.has_edge(self.t, "OTHER_REL", self.o))
        self.assertFalse(self.b.has_edge(self.o, "HAS_OBJECTIVE", self.t))

    def test_create_edge_dedupe(self) -> None:
        self.b.create_edge(self.t, "HAS_OBJECTIVE", self.o)
        self.b.create_edge(self.t, "HAS_OBJECTIVE", self.o)
        self.b.create_edge(self.t, "HAS_OBJECTIVE", self.o)
        # Only one entry should survive in the underlying list AND set.
        self.assertEqual(
            sum(1 for e in self.b._out[self.t]
                if e["rel"] == "HAS_OBJECTIVE" and int(e["dst"]) == self.o),
            1,
        )
        self.assertEqual(
            sum(1 for k in self.b._out_keys[self.t]
                if k == ("HAS_OBJECTIVE", self.o)),
            1,
        )

    def test_delete_evicts_from_set(self) -> None:
        self.b.create_edge(self.t, "HAS_OBJECTIVE", self.o)
        self.b.delete_node(self.o)
        self.assertFalse(self.b.has_edge(self.t, "HAS_OBJECTIVE", self.o))
        self.assertNotIn(("HAS_OBJECTIVE", self.o), self.b._out_keys.get(self.t, set()))

    def test_load_rebuilds_out_keys(self) -> None:
        self.b.create_edge(self.t, "HAS_OBJECTIVE", self.o)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rg.json"
            self.b.save_to_path(path)
            b2 = InMemoryGraphBackend()
            b2.load_from_path(path)
            self.assertTrue(b2.has_edge(self.t, "HAS_OBJECTIVE", self.o))
            self.assertIn(("HAS_OBJECTIVE", self.o), b2._out_keys.get(self.t, set()))


# ---------------------------------------------------------------------------
# Global retrieval index (DF + postings)
# ---------------------------------------------------------------------------


class RetrievalIndexCacheTest(unittest.TestCase):
    """The lazy `_retrieval_index[(kind)]` slot must be:
      - built on first access by walking existing rows
      - incrementally updated by `upsert_embedding` (additions diff)
      - correct under re-index (subtractions diff for stale tokens)
      - drained on doc removal (term frequency drops to 0)
    """

    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(self.backend, tid, "q?")

    def test_first_index_build_populates_slot(self) -> None:
        sid, _ = dedup.upsert_source(
            self.backend, self.oid, "u", "t",
            "alpha beta gamma",
        )
        slot = store._ensure_retrieval_index_built(self.backend, "source")
        self.assertEqual(slot["total_docs"], 1)
        self.assertEqual(slot["df"], {"alpha": 1, "beta": 1, "gamma": 1})
        self.assertEqual(slot["doc_terms"][sid], {"alpha", "beta", "gamma"})
        self.assertIn(sid, slot["postings"]["alpha"])

    def test_incremental_addition(self) -> None:
        s1, _ = dedup.upsert_source(self.backend, self.oid, "u1", "t1", "alpha beta")
        s2, _ = dedup.upsert_source(self.backend, self.oid, "u2", "t2", "beta gamma")
        slot = store._ensure_retrieval_index_built(self.backend, "source")
        self.assertEqual(slot["total_docs"], 2)
        self.assertEqual(slot["df"]["beta"], 2)
        self.assertEqual(slot["df"]["alpha"], 1)
        self.assertEqual(slot["df"]["gamma"], 1)
        self.assertEqual(slot["postings"]["beta"], {s1, s2})

    def test_re_index_subtracts_old_tokens(self) -> None:
        s1, _ = dedup.upsert_source(self.backend, self.oid, "u1", "t1", "alpha beta")
        # Force a re-index with completely different tokens — the
        # store-level upsert_embedding diff path must subtract alpha
        # and beta from the index.
        store.upsert_embedding(self.backend, "source", s1, {"delta": 1, "epsilon": 1})
        slot = store._get_retrieval_index_slot(self.backend, "source")
        self.assertNotIn("alpha", slot["df"])
        self.assertNotIn("beta", slot["df"])
        self.assertEqual(slot["df"]["delta"], 1)
        self.assertEqual(slot["df"]["epsilon"], 1)
        self.assertEqual(slot["postings"].get("alpha"), None)
        self.assertEqual(slot["postings"]["delta"], {s1})


# ---------------------------------------------------------------------------
# search_sources uses the cached IDF + candidate filter
# ---------------------------------------------------------------------------


class SearchUsesIndexCacheTest(unittest.TestCase):
    """Behavioral test: search results must remain stable after the
    iter-3 perf optimisation. Same input → same ranked output."""

    def setUp(self) -> None:
        self.backend = InMemoryGraphBackend()
        tid = store.create_topic(self.backend, "llm")
        self.oid = store.create_objective(self.backend, tid, "q?")
        for url, content in [
            ("u1", "Llama 3 reaches 18 tokens per second on Mac mini metal"),
            ("u2", "Mistral 7B benchmark shows 22 tokens per second under MLX"),
            ("u3", "Linux kernel scheduler analysis under load"),
            ("u4", "Apple Silicon Mac mini local llm performance comparison"),
        ]:
            dedup.upsert_source(self.backend, self.oid, url, "t", content)

    def test_top_result_is_relevant(self) -> None:
        results = retrieval.search_sources(
            self.backend, "llama mac mini", objective_id=self.oid, top_k=5
        )
        self.assertGreater(len(results), 0)
        # The Llama doc must be top result.
        self.assertEqual(results[0]["url"], "u1")

    def test_unrelated_query_filters_out_irrelevant(self) -> None:
        # A query about something completely unrelated to the corpus
        # may still match SOME doc by stop-word coincidence, but the
        # Linux kernel doc is the only one that mentions "scheduler".
        results = retrieval.search_sources(
            self.backend, "linux scheduler", objective_id=self.oid, top_k=5
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["url"], "u3")

    def test_query_filter_skips_obviously_unrelated_docs(self) -> None:
        """The candidate-filter optimisation should NEVER score a doc
        that has zero query-token overlap. We assert this indirectly:
        a query for `kernel` returns only `u3` (the only doc with
        `kernel`), not `u1`/`u2`/`u4`."""
        results = retrieval.search_sources(
            self.backend, "kernel", objective_id=self.oid, top_k=5
        )
        urls = [r["url"] for r in results]
        self.assertEqual(urls, ["u3"])


class SearchPerfRegressionTest(unittest.TestCase):
    """Per-query retrieval cost must stay below a generous ceiling.
    iter-2 baseline at N=2000 was ~9.6 ms/query; iter-3 with cached
    IDF + candidate filter dropped it to ~7.0 ms/query. Ceiling 25 ms
    catches a >3× regression but won't flake on slow CI hosts."""

    def test_query_latency_under_loose_ceiling(self) -> None:
        import random
        random.seed(2026)
        WORDS = [
            "llama", "mistral", "qwen", "phi", "gemma", "metal", "cuda", "mlx",
            "macbook", "macmini", "rtx", "m1", "m2", "m3", "tokens", "second",
            "inference", "context", "window", "quantized", "gguf", "ollama",
        ]
        b = InMemoryGraphBackend()
        tid = store.create_topic(b, "llm")
        oid = store.create_objective(b, tid, "q?")
        N = 800
        for i in range(N):
            content = " ".join(random.sample(WORDS, 8)) + f" doc{i} unique_{i*13}"
            dedup.upsert_source(b, oid, f"u{i}", f"t{i}", content)
        queries = ["llama mac mini metal", "mistral apple", "tokens per second"]
        t0 = time.perf_counter()
        for _ in range(50):
            for q in queries:
                retrieval.search_sources(b, q, objective_id=oid, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_query_ms = elapsed_ms / (50 * len(queries))
        self.assertLess(
            per_query_ms, 25.0,
            f"per-query retrieval cost {per_query_ms:.3f} ms exceeds ceiling — "
            f"IDF cache or candidate filter likely broken"
        )


# ---------------------------------------------------------------------------
# Plugin update / delete commands
# ---------------------------------------------------------------------------


class PluginUpdateDeleteTest(unittest.TestCase):
    """`cmd_update` and `cmd_delete` give OpenClaw CRUQD U/D parity at
    the plugin boundary. Spec L47, L49, L61-68."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="rg-plugin-iter3-")
        self.db = str(Path(self.tmp) / "rg.json")
        plugin_dir = Path(__file__).resolve().parent.parent / "plugins" / "research-memory"
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed(self) -> tuple[int, int, int, int]:
        backend = InMemoryGraphBackend()
        try:
            tid = store.create_topic(backend, "llm")
            oid = store.create_objective(backend, tid, "best local llm")
            sid, _ = dedup.upsert_source(
                backend, oid, "u1", "old title",
                "Llama 3 reaches 18 tokens per second on Apple Silicon Mac mini",
            )
            tid_th = store.insert_thinking(
                backend, oid, "synthesis", "h", [sid], "actor",
            )
            backend.save_to_path(self.db)
            return tid, oid, sid, tid_th
        finally:
            backend.close()

    def _call(self, cmd_name: str, payload: dict) -> dict:
        import argparse
        import plugin_helper as ph  # type: ignore[import-not-found]
        ns = argparse.Namespace(db=self.db, payload=json.dumps(payload))
        fn = getattr(ph, f"cmd_{cmd_name}")
        return fn(ns)

    # --- update -----------------------------------------------------

    def test_update_source_metadata(self) -> None:
        _, _, sid, _ = self._seed()
        out = self._call("update", {
            "kind": "source",
            "id": sid,
            "fields": {"url": "https://updated.example/", "title": "new title"},
        })
        self.assertEqual(out["kind"], "source")
        self.assertEqual(out["id"], sid)
        self.assertEqual(set(out["updated"]), {"url", "title"})
        self.assertEqual(out["row"]["url"], "https://updated.example/")
        self.assertEqual(out["row"]["title"], "new title")
        # Internal _terms must NOT leak.
        self.assertNotIn("_terms", out["row"])

    def test_update_source_rejects_content_mutation(self) -> None:
        _, _, sid, _ = self._seed()
        with self.assertRaises(ValueError):
            self._call("update", {
                "kind": "source",
                "id": sid,
                "fields": {"content": "rewritten content"},
            })

    def test_update_thinking_author(self) -> None:
        _, _, _, tid = self._seed()
        out = self._call("update", {
            "kind": "thinking",
            "id": tid,
            "fields": {"author": "actor-v2"},
        })
        self.assertEqual(out["row"]["author"], "actor-v2")

    def test_update_objective_status(self) -> None:
        _, oid, _, _ = self._seed()
        out = self._call("update", {
            "kind": "objective",
            "id": oid,
            "fields": {"status": "researched"},
        })
        self.assertEqual(out["row"]["status"], "researched")

    def test_update_rejects_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            self._call("update", {"kind": "topic", "id": 1, "fields": {"name": "x"}})

    def test_update_rejects_missing_id(self) -> None:
        with self.assertRaises(ValueError):
            self._call("update", {"kind": "source", "fields": {"url": "x"}})

    def test_update_rejects_empty_fields(self) -> None:
        _, _, sid, _ = self._seed()
        with self.assertRaises(ValueError):
            self._call("update", {"kind": "source", "id": sid, "fields": {}})

    # --- delete -----------------------------------------------------

    def test_delete_source(self) -> None:
        _, _, sid, _ = self._seed()
        out = self._call("delete", {"kind": "source", "id": sid})
        self.assertEqual(out, {"kind": "source", "id": sid, "deleted": True})
        # Verify it's gone.
        backend = InMemoryGraphBackend()
        backend.load_from_path(self.db)
        try:
            self.assertIsNone(store.get_source(backend, sid))
        finally:
            backend.close()

    def test_delete_thinking(self) -> None:
        _, _, _, tid = self._seed()
        out = self._call("delete", {"kind": "thinking", "id": tid})
        self.assertEqual(out["deleted"], True)

    def test_delete_objective_cascades(self) -> None:
        _, oid, sid, tid = self._seed()
        out = self._call("delete", {"kind": "objective", "id": oid})
        self.assertEqual(out["deleted"], True)
        backend = InMemoryGraphBackend()
        backend.load_from_path(self.db)
        try:
            self.assertIsNone(store.get_objective(backend, oid))
            # Children should be cascaded.
            self.assertIsNone(store.get_source(backend, sid))
            self.assertIsNone(store.get_thinking(backend, tid))
        finally:
            backend.close()

    def test_delete_unknown_id_raises(self) -> None:
        self._seed()
        with self.assertRaises(KeyError):
            self._call("delete", {"kind": "source", "id": 99999})

    def test_delete_rejects_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            self._call("delete", {"kind": "garbage", "id": 1})


if __name__ == "__main__":
    unittest.main()
