"""Graph backend subpackage: label/relationship constants + pluggable backends.

`Node` / `Edge` / `NodeRef` dataclasses were removed in iteration 2 —
they had zero constructor calls in the entire codebase. See
`research_graph.graph.model` docstring for rationale (spec L73-74).
"""
from .model import (
    LABEL_TOPIC,
    LABEL_OBJECTIVE,
    LABEL_SOURCE,
    LABEL_THINKING,
    REL_HAS_OBJECTIVE,
    REL_HAS_SOURCE,
    REL_HAS_THINKING,
    REL_CITES,
    REL_SUPERSEDES,
    REL_REUSES,
)
from .backend import GraphBackend
from .inmemory import InMemoryGraphBackend
from .factory import get_default_backend

__all__ = [
    "LABEL_TOPIC",
    "LABEL_OBJECTIVE",
    "LABEL_SOURCE",
    "LABEL_THINKING",
    "REL_HAS_OBJECTIVE",
    "REL_HAS_SOURCE",
    "REL_HAS_THINKING",
    "REL_CITES",
    "REL_SUPERSEDES",
    "REL_REUSES",
    "GraphBackend",
    "InMemoryGraphBackend",
    "get_default_backend",
]
