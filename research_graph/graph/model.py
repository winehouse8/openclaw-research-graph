"""Canonical node / edge labels shared by every graph backend.

The graph backends return flat dicts (`{"id": ..., "label": ...,
"properties": ...}`) rather than typed `Node` / `Edge` instances, so
the dataclasses that previously lived here were dead code (zero
constructor calls anywhere in `research_graph/`, `tests/`, or
`plugins/`). Keeping them as exports gave new contributors a fake
"typed model" to chase that doesn't actually exist in the codebase.

Removed in iteration 2 (spec L73-74 anti-over-engineering):
  - `Node` dataclass
  - `Edge` dataclass
  - `NodeRef` frozen dataclass

The label / relationship constants below are the only API this
module exposes now.
"""
from __future__ import annotations


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


# Note: Node / Edge / NodeRef dataclasses removed in iteration 2.
# See module docstring above for rationale.
