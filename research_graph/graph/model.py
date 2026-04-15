"""Canonical node / edge model shared by every graph backend.

The goal of this module is that a caller working against the interface in
``backend.py`` never has to know whether it is talking to the pure-python
in-memory graph or a real Neo4j instance -- the node and edge shapes are
the same in both worlds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------

LABEL_TOPIC = "Topic"
LABEL_OBJECTIVE = "Objective"
LABEL_SOURCE = "Source"
LABEL_THINKING = "Thinking"

NODE_LABELS = (LABEL_TOPIC, LABEL_OBJECTIVE, LABEL_SOURCE, LABEL_THINKING)


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

REL_HAS_OBJECTIVE = "HAS_OBJECTIVE"    # (Topic)-[:HAS_OBJECTIVE]->(Objective)
REL_HAS_SOURCE = "HAS_SOURCE"          # (Objective)-[:HAS_SOURCE]->(Source)
REL_HAS_THINKING = "HAS_THINKING"      # (Objective)-[:HAS_THINKING]->(Thinking)
REL_CITES = "CITES"                    # (Thinking)-[:CITES]->(Source)
REL_SUPERSEDES = "SUPERSEDES"          # (Thinking)-[:SUPERSEDES]->(Thinking)
REL_REUSES = "REUSES"                  # (Thinking)-[:REUSES]->(Thinking)

RELATIONSHIP_TYPES = (
    REL_HAS_OBJECTIVE,
    REL_HAS_SOURCE,
    REL_HAS_THINKING,
    REL_CITES,
    REL_SUPERSEDES,
    REL_REUSES,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A labelled vertex with a property bag.

    ``id`` is assigned by the backend on creation. Property dicts are
    shallow-copied defensively when crossing backend boundaries so callers
    cannot accidentally mutate stored state.
    """

    id: int
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the flat dict representation used throughout the code.

        Domain code expects to access node fields as ``row["id"]``,
        ``row["name"]``, ``row["url"]``, etc. We flatten the property
        bag into that shape and inject ``id`` / ``label`` so consumers
        still have typing information available.
        """
        out: dict[str, Any] = dict(self.properties)
        out["id"] = int(self.id)
        out["label"] = self.label
        return out


@dataclass
class Edge:
    """A typed directed edge between two nodes."""

    src_id: int
    rel: str
    dst_id: int
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeRef:
    """A lightweight reference to a node without loading its properties."""

    id: int
    label: str
