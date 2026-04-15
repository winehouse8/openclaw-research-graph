from __future__ import annotations

from pathlib import Path

from .graph import get_default_backend
from .orchestrator import Orchestrator


def research(db_path: str | Path, objective_id: int, force_refresh: bool = False) -> dict:
    backend = get_default_backend(db_path)
    try:
        orch = Orchestrator(backend)
        result = orch.research(objective_id, force_refresh=force_refresh)
        return result.to_dict()
    finally:
        backend.close()
