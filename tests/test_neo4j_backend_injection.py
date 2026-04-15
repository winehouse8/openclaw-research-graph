"""HIGH 1 regression: Cypher label/rel/property-key allowlists.

These tests exercise the module-level allowlist helpers directly (no
live Neo4j connection needed) and also drive the Neo4jBackend public
read / traversal methods with a mocked driver, asserting that
malicious inputs raise ``ValueError`` BEFORE any Cypher statement is
dispatched to the driver.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from research_graph.graph.neo4j_backend import (
    Neo4jBackend,
    _VALID_LABELS,
    _VALID_PROP_KEY,
    _VALID_RELS,
    _check_label,
    _check_prop_key,
    _check_rel,
    _where_clause,
)


class AllowlistHelperTest(unittest.TestCase):
    def test_valid_label_passes(self) -> None:
        for lbl in ("Topic", "Objective", "Source", "Thinking"):
            _check_label(lbl)  # no raise
            self.assertIn(lbl, _VALID_LABELS)

    def test_malicious_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _check_label("Topic) DETACH DELETE n //")
        with self.assertRaises(ValueError):
            _check_label("")
        with self.assertRaises(ValueError):
            _check_label("topic")  # case-sensitive

    def test_valid_rel_passes(self) -> None:
        for rel in ("HAS_OBJECTIVE", "HAS_SOURCE", "HAS_THINKING",
                    "CITES", "SUPERSEDES", "REUSES"):
            _check_rel(rel)
            self.assertIn(rel, _VALID_RELS)

    def test_malicious_rel_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _check_rel("CITES] DETACH DELETE")
        with self.assertRaises(ValueError):
            _check_rel("CITES|DELETE")
        with self.assertRaises(ValueError):
            _check_rel("")

    def test_valid_prop_key_passes(self) -> None:
        for k in ("name", "id", "_terms", "content_hash", "a1_b2"):
            _check_prop_key(k)
            self.assertIsNotNone(_VALID_PROP_KEY.match(k))

    def test_malicious_prop_key_rejected(self) -> None:
        for bad in ("name} RETURN n //", "na me", "1name", "n.k", "n:k", ""):
            with self.assertRaises(ValueError):
                _check_prop_key(bad)

    def test_where_clause_rejects_bad_key(self) -> None:
        with self.assertRaises(ValueError):
            _where_clause({"name} RETURN n //": "x"})

    def test_where_clause_roundtrip(self) -> None:
        clause, params = _where_clause({"name": "alpha", "id": 7})
        self.assertTrue(clause.startswith("WHERE "))
        self.assertIn("n.name = $w_name", clause)
        self.assertIn("n.id = $w_id", clause)
        self.assertEqual(params, {"w_name": "alpha", "w_id": 7})


class Neo4jBackendMockedMethodTest(unittest.TestCase):
    """Instantiate Neo4jBackend via __new__ (bypassing __init__ so no
    driver is imported or connected) and wire a MagicMock driver. Every
    malicious input must raise before the driver is touched."""

    def _make(self) -> Neo4jBackend:
        obj = Neo4jBackend.__new__(Neo4jBackend)
        obj._driver = MagicMock(name="driver")  # type: ignore[attr-defined]
        obj._database = None  # type: ignore[attr-defined]
        return obj

    def _assert_driver_untouched(self, obj: Neo4jBackend) -> None:
        driver = obj._driver  # type: ignore[attr-defined]
        self.assertEqual(
            driver.session.call_count, 0,
            "driver.session() must not be called on malicious input",
        )

    # -- find_node ---------------------------------------------------------
    def test_find_node_rejects_bad_label(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.find_node("Topic) DETACH DELETE n //", {"name": "x"})
        self._assert_driver_untouched(obj)

    def test_find_node_rejects_bad_prop_key(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.find_node("Topic", {"name} RETURN n //": "x"})
        self._assert_driver_untouched(obj)

    # -- list_nodes --------------------------------------------------------
    def test_list_nodes_rejects_bad_label(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.list_nodes("Topic) DETACH DELETE n //")
        self._assert_driver_untouched(obj)

    def test_list_nodes_rejects_bad_prop_key(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.list_nodes("Topic", {"name} RETURN": "x"})
        self._assert_driver_untouched(obj)

    # -- out_neighbors -----------------------------------------------------
    def test_out_neighbors_rejects_bad_rel(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.out_neighbors(1, rel="CITES] DETACH DELETE")
        self._assert_driver_untouched(obj)

    def test_out_neighbors_rejects_bad_label(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.out_neighbors(1, label="Topic) //")
        self._assert_driver_untouched(obj)

    # -- in_neighbors ------------------------------------------------------
    def test_in_neighbors_rejects_bad_rel(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.in_neighbors(1, rel="BOGUS")
        self._assert_driver_untouched(obj)

    def test_in_neighbors_rejects_bad_label(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.in_neighbors(1, label="NotALabel")
        self._assert_driver_untouched(obj)

    # -- shortest_path -----------------------------------------------------
    def test_shortest_path_rejects_bad_rel_in_list(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.shortest_path(1, 2, rels=["CITES", "CITES] DETACH"])
        self._assert_driver_untouched(obj)

    # -- neighborhood ------------------------------------------------------
    def test_neighborhood_rejects_bad_rel_in_list(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.neighborhood(1, 2, rels=["BOGUS"])
        self._assert_driver_untouched(obj)

    # -- create_node / create_edge reuse the same allowlist --------------
    def test_create_node_rejects_bad_label(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.create_node("Topic) //", {"name": "x"})
        self._assert_driver_untouched(obj)

    def test_create_node_rejects_bad_prop_key(self) -> None:
        obj = self._make()
        # _next_id would hit the driver; the key check must run first.
        with self.assertRaises(ValueError):
            obj.create_node("Topic", {"name} RETURN": "x"})
        self._assert_driver_untouched(obj)

    def test_create_edge_rejects_bad_rel(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.create_edge(1, "CITES] DELETE", 2)
        self._assert_driver_untouched(obj)

    # -- update_node -------------------------------------------------------
    def test_update_node_rejects_bad_prop_key(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.update_node(1, {"k} SET n.x = 1 //": "v"})
        self._assert_driver_untouched(obj)

    # -- has_edge ----------------------------------------------------------
    def test_has_edge_rejects_bad_rel(self) -> None:
        obj = self._make()
        with self.assertRaises(ValueError):
            obj.has_edge(1, "NOPE", 2)
        self._assert_driver_untouched(obj)


if __name__ == "__main__":
    unittest.main()
