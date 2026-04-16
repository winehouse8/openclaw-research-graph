"""Evaluation layer — run log + quality-over-time trend.

iter-7: before this module, individual Thinkings had quality_overall
(iter-4) and the orchestrator gated supersession on quality delta, but
nobody recorded WHETHER a daily research cycle actually improved the
system. The spec says "매일 자동으로 리서치 → 기존 답을 더 나은 방향
으로 업데이트" — without a run log, that claim is unverifiable.

This module provides:

  * `record_run(log_path, journey_result)` — append one journey
    result (the dict from `api.research_journey`) to a JSONL file.
    One line per run, timestamped, with the quality-relevant fields
    extracted for fast trend queries. The raw journey dict is stored
    too (under `_raw`) so nothing is lost.

  * `quality_trend(log_path, objective_id, last_n)` — read the last
    N entries for an objective and return a trend summary: list of
    (timestamp, quality_overall, outcome) tuples + a monotonicity
    flag (`improved_or_stable`) + the overall delta from first to
    last entry. This is what the plugin surfaces as "is this topic
    getting better?"

  * `stale_objectives(log_path, topic_name, max_age_hours)` — find
    objectives that haven't been researched in the last N hours.
    Useful for a scheduler that wants to prioritize stale topics.

The log file is a sibling of the graph database file:
  `<db_dir>/research_runs.jsonl`
This keeps the log co-located with the graph so a backup/move of
the db directory captures both.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_log_path(db_path: str | Path) -> Path:
    """Derive the run-log path from the database path."""
    db = Path(db_path)
    return db.parent / "research_runs.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_run(
    log_path: str | Path | None,
    journey_result: dict,
    *,
    db_path: str | Path | None = None,
) -> dict:
    """Append one journey result to the JSONL run log.

    Either `log_path` or `db_path` must be provided. If `log_path`
    is None, it's derived from `db_path` via `_default_log_path`.

    Returns the entry dict that was written (for inspection / tests).
    """
    if log_path is None:
        if db_path is None:
            raise ValueError("either log_path or db_path must be provided")
        log_path = _default_log_path(db_path)
    log_path = Path(log_path)

    # Extract the quality-relevant fields into a flat entry for fast
    # trend queries without re-parsing the full journey dict.
    delta = journey_result.get("delta") or {}
    objective = journey_result.get("objective") or {}
    topic = journey_result.get("topic") or {}
    after = journey_result.get("memory_after") or {}
    run = journey_result.get("run") or {}

    entry = {
        "timestamp": _now_iso(),
        "topic_id": topic.get("id"),
        "topic_name": topic.get("name"),
        "objective_id": objective.get("id"),
        "question": objective.get("question"),
        "outcome": delta.get("outcome"),
        "quality_overall": after.get("live_quality_overall"),
        "quality_delta": delta.get("quality_overall_delta"),
        "live_thinking_id": after.get("live_thinking_id"),
        "source_count": after.get("source_count"),
        "thinking_count": after.get("thinking_count"),
        "live_count": after.get("live_count", 0),
        "sibling_count": len(after.get("sibling_live_thinkings") or []),
        "new_source_count": len(run.get("new_source_ids") or []),
        "actor_backend": run.get("actor_backend"),
        "narrative": delta.get("narrative"),
        "_raw": journey_result,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    return entry


def _read_log(log_path: str | Path) -> list[dict]:
    """Read all entries from the JSONL log. Returns [] if file doesn't exist."""
    p = Path(log_path)
    if not p.exists():
        return []
    entries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def quality_trend(
    log_path: str | Path | None,
    objective_id: int,
    last_n: int = 20,
    *,
    db_path: str | Path | None = None,
) -> dict:
    """Return the quality trend for an objective from the run log.

    Returns:
      {
        "objective_id": int,
        "entries": [ {timestamp, quality_overall, outcome, ...}, ... ],
        "run_count": int,
        "first_quality": float | None,
        "latest_quality": float | None,
        "overall_delta": float | None,
        "improved_or_stable": bool,  # True if quality never dropped
        "improvement_count": int,    # number of runs where quality went up
        "regression_count": int,     # number of runs where quality went down
      }
    """
    if log_path is None:
        if db_path is None:
            raise ValueError("either log_path or db_path must be provided")
        log_path = _default_log_path(db_path)

    all_entries = _read_log(log_path)
    # Filter to this objective and take last_n.
    relevant = [
        e for e in all_entries
        if e.get("objective_id") == int(objective_id)
    ]
    if last_n and len(relevant) > last_n:
        relevant = relevant[-last_n:]

    # Compute trend stats.
    qualities = [
        e.get("quality_overall")
        for e in relevant
        if e.get("quality_overall") is not None
    ]
    first_q = qualities[0] if qualities else None
    latest_q = qualities[-1] if qualities else None
    overall_delta = (
        round(float(latest_q) - float(first_q), 4)
        if first_q is not None and latest_q is not None
        else None
    )

    improved_or_stable = True
    improvement_count = 0
    regression_count = 0
    for i in range(1, len(qualities)):
        delta = qualities[i] - qualities[i - 1]
        if delta > 0.001:
            improvement_count += 1
        elif delta < -0.001:
            regression_count += 1
            improved_or_stable = False

    # Slim the entries for the response (drop _raw to keep it light).
    slim = []
    for e in relevant:
        slim.append({
            "timestamp": e.get("timestamp"),
            "quality_overall": e.get("quality_overall"),
            "outcome": e.get("outcome"),
            "quality_delta": e.get("quality_delta"),
            "live_thinking_id": e.get("live_thinking_id"),
            "actor_backend": e.get("actor_backend"),
            "narrative": e.get("narrative"),
        })

    return {
        "objective_id": int(objective_id),
        "run_count": len(relevant),
        "entries": slim,
        "first_quality": first_q,
        "latest_quality": latest_q,
        "overall_delta": overall_delta,
        "improved_or_stable": improved_or_stable,
        "improvement_count": improvement_count,
        "regression_count": regression_count,
    }


def stale_objectives(
    log_path: str | Path | None,
    topic_name: str | None = None,
    max_age_hours: float = 24.0,
    *,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Find objectives not researched in the last `max_age_hours`.

    Returns a list of {objective_id, question, topic_name,
    last_run_timestamp, hours_since_last_run} sorted by staleness
    descending. If `topic_name` is given, only objectives under that
    topic are returned.
    """
    if log_path is None:
        if db_path is None:
            raise ValueError("either log_path or db_path must be provided")
        log_path = _default_log_path(db_path)

    all_entries = _read_log(log_path)
    now = datetime.now(timezone.utc)

    # Build a map: objective_id → latest entry.
    latest: dict[int, dict] = {}
    for e in all_entries:
        oid = e.get("objective_id")
        if oid is None:
            continue
        if topic_name and e.get("topic_name") != topic_name:
            continue
        prev = latest.get(oid)
        if prev is None or (e.get("timestamp") or "") > (prev.get("timestamp") or ""):
            latest[oid] = e

    stale = []
    for oid, entry in latest.items():
        ts_str = entry.get("timestamp") or ""
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            # Can't parse → treat as ancient.
            hours = max_age_hours + 1
            ts = None
        else:
            hours = (now - ts).total_seconds() / 3600

        if hours >= max_age_hours:
            stale.append({
                "objective_id": oid,
                "question": entry.get("question"),
                "topic_name": entry.get("topic_name"),
                "last_run_timestamp": ts_str,
                "hours_since_last_run": round(hours, 2),
            })

    stale.sort(key=lambda s: -s["hours_since_last_run"])
    return stale
