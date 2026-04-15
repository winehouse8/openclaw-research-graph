from __future__ import annotations

from pathlib import Path

from . import storage
from .orchestrator import Orchestrator


def research(db_path: str | Path, objective_id: int, force_refresh: bool = False) -> dict:
    conn = storage.connect(db_path)
    try:
        orch = Orchestrator(conn)
        result = orch.research(objective_id, force_refresh=force_refresh)
        obj = storage.get_objective(conn, objective_id)
        return {
            "objective": obj,
            "mode": result.mode,
            "new_sources": result.new_source_ids,
            "reused_sources": result.reused_source_ids,
            "new_thinking": result.new_thinking_id,
            "supersedes": result.supersedes_id,
            "rejected_reasons": result.rejected_reasons,
        }
    finally:
        conn.close()
