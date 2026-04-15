"""Graph backend subpackage: canonical node/edge model + pluggable backends."""
from .model import (
    Edge,
    Node,
    NodeRef,
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
    "Edge",
    "Node",
    "NodeRef",
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
