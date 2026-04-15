from __future__ import annotations

from pathlib import Path

from . import storage
from .orchestrator import Orchestrator


def research(db_path: str | Path, objective_id: int, force_refresh: bool = False) -> dict:
    conn = storage.connect(db_path)
    try:
        orch = Orchestrator(conn)
        result = orch.research(objective_id, force_refresh=force_refresh)
        return result.to_dict()
    finally:
        conn.close()
