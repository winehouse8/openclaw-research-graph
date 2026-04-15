from __future__ import annotations

import sys
from pathlib import Path

from . import storage
from .orchestrator import Orchestrator


def cron_line(hour: int, topic: str, db_path: str) -> str:
    py = sys.executable or "python3"
    return (
        f"0 {hour} * * * {py} -m research_graph schedule once "
        f"--topic {topic!r} --db {db_path!r}"
    )


def run_once_for_topic(db_path: str | Path, topic_name: str) -> list[dict]:
    conn = storage.connect(db_path)
    try:
        topic = storage.get_topic_by_name(conn, topic_name)
        if topic is None:
            return []
        orch = Orchestrator(conn)
        out = []
        for obj in storage.query_objectives(conn, topic_id=int(topic["id"])):
            res = orch.research(int(obj["id"]))
            out.append(
                {
                    "objective_id": int(obj["id"]),
                    "mode": res.mode,
                    "new_sources": res.new_source_ids,
                    "new_thinking": res.new_thinking_id,
                }
            )
        return out
    finally:
        conn.close()
