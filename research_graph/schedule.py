"""Scheduler — daily continual-research entry point.

iter-7 rewrite: the pre-iter-7 scheduler was 39 lines that called
`Orchestrator(backend).research()` directly, throwing away:
  - the before/after memory snapshot (no delta tracking)
  - quality score comparison (no way to know if runs improve)
  - the journey narrative (no summary for the cron operator)
  - the evaluation run log (no quality-over-time tracking)

After day 1, every cron run was a no-op because dedup caught
the same fixture sources. The scheduler returned a minimal dict
that told the operator nothing about whether quality improved.

This rewrite routes through `api.research_journey` (the same path
the plugin uses) so every scheduled run:
  1. Resolves or creates the (topic, objective) pair
  2. Snapshots memory before/after
  3. Runs the quality-gated orchestrator
  4. Records the journey result to the evaluation run log
  5. Returns the full journey payload with delta/narrative

The `cron_line` helper is kept for generating a crontab entry.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import api, evaluation, store
from .graph import get_default_backend


def cron_line(hour: int, topic: str, db_path: str) -> str:
    """Generate a crontab line for a daily research run."""
    py = sys.executable or "python3"
    return (
        f"0 {hour} * * * {py} -m research_graph schedule once "
        f"--topic {topic!r} --db {db_path!r}"
    )


def run_once_for_topic(
    db_path: str | Path,
    topic_name: str,
    *,
    skip_external: bool = False,
) -> list[dict]:
    """Run one research cycle for every objective under `topic_name`.

    This is the main cron entry point. For each objective it:
      1. Calls `api.research_journey` (before/after snapshot + quality
         gate + delta classification)
      2. Records the journey result to the evaluation run log
      3. Collects the journey results into a list

    Returns a list of journey dicts (one per objective). The caller
    (CLI or cron wrapper) can inspect `delta.outcome` and
    `delta.quality_overall_delta` to decide whether to alert.
    """
    backend = get_default_backend(db_path)
    try:
        topic = store.get_topic_by_name(backend, topic_name)
        if topic is None:
            return []
        objectives = store.query_objectives(backend, topic_id=int(topic["id"]))
        questions = [(int(obj["id"]), obj.get("question", "")) for obj in objectives]
    finally:
        backend.close()

    if not questions:
        return []

    log_path = evaluation._default_log_path(db_path)
    results = []
    for _oid, question in questions:
        if not question:
            continue
        journey = api.research_journey(
            db_path,
            topic_name,
            question,
            skip_external=skip_external,
        )
        evaluation.record_run(log_path, journey)
        results.append(journey)
    return results


def run_daily_summary(
    db_path: str | Path,
    topic_name: str,
    *,
    skip_external: bool = False,
) -> dict:
    """Run all objectives + return an aggregate daily summary.

    This is the "매일 오전 9시" entry point that returns a single dict
    the operator can glance at to know whether the system is healthy:

      {
        "topic": str,
        "objectives_run": int,
        "outcomes": {"cold_start": N, "improved": N, "reused": N, ...},
        "quality_deltas": [float, ...],    # one per objective
        "all_stable_or_improved": bool,
        "journeys": [ ... full journey dicts ... ],
      }
    """
    journeys = run_once_for_topic(
        db_path, topic_name, skip_external=skip_external
    )
    outcomes: dict[str, int] = {}
    deltas: list[float | None] = []
    for j in journeys:
        d = j.get("delta", {})
        out = d.get("outcome", "unknown")
        outcomes[out] = outcomes.get(out, 0) + 1
        deltas.append(d.get("quality_overall_delta"))

    numeric_deltas = [d for d in deltas if d is not None]
    all_stable = all(d >= -0.001 for d in numeric_deltas) if numeric_deltas else True

    return {
        "topic": topic_name,
        "objectives_run": len(journeys),
        "outcomes": outcomes,
        "quality_deltas": deltas,
        "all_stable_or_improved": all_stable,
        "journeys": journeys,
    }
