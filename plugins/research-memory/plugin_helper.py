"""
Plugin-side Python helper for the openclaw research-memory plugin.

This script is executed as a subprocess from the JS plugin entry. It bridges
operations that are not directly exposed by `python3 -m research_graph` JSON
output (status aggregation, direct source ingestion, topic+objective resolve
+ research in one call) while reusing the same `research_graph` Python
package so there is exactly one canonical memory store.

All subcommands print one JSON document to stdout and ALWAYS exit 0. Failure
is signalled in-band via {"ok": false, "error", "type", "traceback"} so the
JS caller can parse a structured error object instead of getting a flattened
stderr string. Success payloads omit the "ok" key (callers treat absent ok
as success for backwards compat).

Intentionally stdlib-only so it runs wherever the `research_graph` package
runs (no extra deps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Make the sibling `research_graph` package importable no matter where the
# plugin is installed. Precedence:
#   1. RESEARCH_GRAPH_PACKAGE_DIR env var (explicit override)
#   2. Two directories up from this file (host layout: repo/plugins/research-memory/)
#   3. Whatever the active sys.path already resolves
_env_pkg = os.environ.get("RESEARCH_GRAPH_PACKAGE_DIR")
if _env_pkg:
    sys.path.insert(0, _env_pkg)
else:
    _guess = Path(__file__).resolve().parent.parent.parent
    if (_guess / "research_graph" / "__init__.py").is_file():
        sys.path.insert(0, str(_guess))

from research_graph import api as rg_api  # noqa: E402
from research_graph import dedup as rg_dedup  # noqa: E402
from research_graph import retrieval as rg_retrieval  # noqa: E402
from research_graph import store as rg_store  # noqa: E402
from research_graph.graph import get_default_backend  # noqa: E402


def _strip_terms(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        c = dict(r)
        c.pop("_terms", None)
        c.pop("label", None)
        out.append(c)
    return out


def _count_label(backend, label: str) -> int:
    try:
        rows = backend.list_nodes(label)
    except Exception:
        rows = []
    return len(rows)


def _count_thinking_rel(backend, rel: str) -> int:
    """Count outbound rel edges from all thinking nodes.

    The GraphBackend protocol exposes only per-node neighbor walks, not a
    global edge scan, so we fan out from every thinking. SUPERSEDES and
    REUSES are both thinking->thinking, so this is exact.
    """
    try:
        thinkings = backend.list_nodes(rg_store.LABEL_THINKING)
    except Exception:
        return 0
    total = 0
    for th in thinkings:
        try:
            peers = backend.out_neighbors(
                int(th["id"]), rel=rel, label=rg_store.LABEL_THINKING
            )
        except Exception:
            peers = []
        total += len(peers)
    return total


def cmd_status(args: argparse.Namespace) -> dict:
    backend = get_default_backend(args.db)
    try:
        counts = {
            "topics": _count_label(backend, rg_store.LABEL_TOPIC),
            "objectives": _count_label(backend, rg_store.LABEL_OBJECTIVE),
            "sources": _count_label(backend, rg_store.LABEL_SOURCE),
            "thinkings": _count_label(backend, rg_store.LABEL_THINKING),
            "supersessions": _count_thinking_rel(backend, rg_store.REL_SUPERSEDES),
            "reuses": _count_thinking_rel(backend, rg_store.REL_REUSES),
        }
        return {"db": args.db, "counts": counts}
    finally:
        backend.close()


def cmd_ingest(args: argparse.Namespace) -> dict:
    payload = json.loads(args.payload)
    objective_id = int(payload["objective_id"])
    sources: list[dict] = payload.get("sources", [])
    backend = get_default_backend(args.db)
    inserted: list[int] = []
    skipped: list[dict] = []
    try:
        for s in sources:
            if not isinstance(s, dict):
                raise TypeError(
                    f"each source must be an object, got {type(s).__name__}"
                )
            if "content" not in s or s.get("content") is None:
                raise ValueError(
                    "source is missing required string field 'content'"
                )
            url = s.get("url")
            title = s.get("title")
            content = s.get("content") or ""
            if not isinstance(content, str):
                raise TypeError(
                    f"source 'content' must be a string, got {type(content).__name__}"
                )
            if not content:
                skipped.append({"url": url, "reason": "empty-content"})
                continue
            # Delegate to dedup.upsert_source so externally-ingested
            # content goes through the same exact-hash + near-duplicate
            # (Jaccard >= threshold) dedup path that internal research
            # runs use. Previously this hand-rolled the insert and
            # skipped near-dup detection — two near-identical payloads
            # ingested via the plugin would create two rows, violating
            # spec L50 (중복 감지). dedup.upsert_source also handles
            # the retrieval index update internally.
            sid, created = rg_dedup.upsert_source(
                backend,
                objective_id=objective_id,
                url=url,
                title=title,
                content=content,
                search_query=None,
            )
            if created:
                inserted.append(int(sid))
            else:
                skipped.append({"url": url, "reason": "duplicate",
                                "source_id": int(sid)})
        return {
            "objective_id": objective_id,
            "inserted": inserted,
            "skipped": skipped,
        }
    finally:
        backend.close()


def cmd_research_topic(args: argparse.Namespace) -> dict:
    """Resolve (topic, question) -> objective_id, then run one research cycle.

    Creates the topic and objective if they do not already exist. Matching is
    by exact topic name and exact objective question text within that topic
    (the same lookup store.get_topic_by_name and query_objectives use).
    """
    topic_name = args.topic
    question = args.question

    backend = get_default_backend(args.db)
    try:
        topic = rg_store.get_topic_by_name(backend, topic_name)
        if topic is None:
            topic_id = rg_store.create_topic(backend, topic_name)
        else:
            topic_id = int(topic["id"])

        objective_id: int | None = None
        for row in rg_store.query_objectives(backend, topic_id=topic_id):
            if row.get("question") == question:
                objective_id = int(row["id"])
                break
        if objective_id is None:
            objective_id = rg_store.create_objective(backend, topic_id, question)
    finally:
        backend.close()

    # api.research opens/closes its own backend so the in-memory JSON backend
    # flushes between the two phases.
    research_result = rg_api.research(args.db, objective_id,
                                      force_refresh=bool(args.force_refresh))
    return {
        "topic": topic_name,
        "topic_id": int(topic_id),
        "objective_id": int(objective_id),
        "question": question,
        "research": research_result,
    }


def cmd_query(args: argparse.Namespace) -> dict:
    """Read accumulated knowledge for a topic without running new research.

    Spec L40 + L61-68: OpenClaw must be able to "관련 지식을 읽고 …
    이후 reasoning 에서 다시 재사용." Before this command, the plugin
    only exposed `status` (counts), `ingest` (write), and
    `research-topic` (write+read entangled). There was no read-only
    surface for "what do you already know about X?" — OpenClaw had to
    invoke the CLI separately and parse JSON, which is awkward and
    couples OpenClaw to the CLI shape.

    Inputs (JSON payload via --payload, mirroring `cmd_ingest`):
      {
        "topic_name": "llm" | "topic_id": 7,        # one of the two
        "question":   "best local llm",             # the query string
        "kind":       "source" | "thinking" | "both",  # default "both"
        "objective_id": 12,                          # optional scope
        "top_k":     5,                              # default 5
        "min_score": 0.0                             # default 0.0
      }

    Returns:
      {
        "topic_id": int | None,
        "topic_name": str | None,
        "objective_id": int | None,
        "kind": "source" | "thinking" | "both",
        "sources":   [ {id, url, title, content_snippet, score}, ... ],
        "thinkings": [ {id, content_snippet, score, supports_source_ids}, ... ]
      }

    The command is read-only: it never writes to the graph, never
    fetches external sources, and never invokes the orchestrator.
    """
    payload = json.loads(args.payload)
    question = payload.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("payload missing required string field 'question'")
    kind = payload.get("kind") or "both"
    if kind not in ("source", "thinking", "both"):
        raise ValueError(f"kind must be one of source|thinking|both, got {kind!r}")
    top_k = int(payload.get("top_k") or 5)
    min_score = float(payload.get("min_score") or 0.0)
    objective_id = payload.get("objective_id")
    if objective_id is not None:
        objective_id = int(objective_id)

    backend = get_default_backend(args.db)
    try:
        topic_id: int | None = None
        topic_name: str | None = payload.get("topic_name")
        if "topic_id" in payload and payload["topic_id"] is not None:
            topic_id = int(payload["topic_id"])
            row = rg_store.get_node(backend, topic_id) if hasattr(rg_store, "get_node") else None
            # Fall back to direct backend lookup for the name when the
            # store doesn't expose a Topic getter.
            if row is None:
                t = backend.get_node(topic_id)
                if t is not None:
                    topic_name = t.get("properties", {}).get("name") or topic_name
        elif topic_name:
            topic = rg_store.get_topic_by_name(backend, topic_name)
            if topic is not None:
                topic_id = int(topic["id"])

        sources_out: list[dict] = []
        thinkings_out: list[dict] = []
        if kind in ("source", "both"):
            rows = rg_retrieval.search_sources(
                backend, question,
                objective_id=objective_id,
                topic_id=topic_id,
                top_k=top_k,
                min_score=min_score,
            )
            for r in rows:
                content = r.get("content") or ""
                sources_out.append({
                    "id": int(r["id"]),
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "content_snippet": content[:280],
                    "score": float(r.get("score") or 0.0),
                })
        if kind in ("thinking", "both"):
            rows = rg_retrieval.search_thinkings(
                backend, question,
                objective_id=objective_id,
                topic_id=topic_id,
                top_k=top_k,
                min_score=min_score,
            )
            for r in rows:
                content = r.get("content") or ""
                thinkings_out.append({
                    "id": int(r["id"]),
                    "content_snippet": content[:280],
                    "score": float(r.get("score") or 0.0),
                    "supports_source_ids": list(r.get("supports_source_ids") or []),
                    "author": r.get("author"),
                })
        return {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "objective_id": objective_id,
            "kind": kind,
            "sources": sources_out,
            "thinkings": thinkings_out,
        }
    finally:
        backend.close()


def cmd_update(args: argparse.Namespace) -> dict:
    """Update mutable metadata on a Source / Thinking / Objective row.

    Spec L47 (CRUQD U) for plugin parity. Routes to
    `store.update_source` / `update_thinking` / `update_objective`,
    each of which has its own immutable allowlist enforced at the
    store layer (content/content_hash NEVER mutable here — content
    edits go through `dedup.upsert_*` which atomically re-indexes).

    Payload shape:
      {
        "kind":   "source" | "thinking" | "objective",
        "id":     int,
        "fields": {field_name: value, ...}
      }

    Returns:
      {"kind": ..., "id": ..., "updated": [field_names], "row": <new row>}
    """
    payload = json.loads(args.payload)
    kind = payload.get("kind")
    if kind not in ("source", "thinking", "objective"):
        raise ValueError(f"kind must be source|thinking|objective, got {kind!r}")
    if "id" not in payload:
        raise ValueError("payload missing required field 'id'")
    target_id = int(payload["id"])
    fields = payload.get("fields") or {}
    if not isinstance(fields, dict) or not fields:
        raise ValueError("payload missing/empty 'fields' dict")

    backend = get_default_backend(args.db)
    try:
        if kind == "source":
            rg_store.update_source(backend, target_id, **fields)
            row = rg_store.get_source(backend, target_id)
        elif kind == "thinking":
            rg_store.update_thinking(backend, target_id, **fields)
            row = rg_store.get_thinking(backend, target_id)
        else:
            rg_store.update_objective(backend, target_id, **fields)
            row = rg_store.get_objective(backend, target_id)
        if row is None:
            raise KeyError(f"unknown {kind}: {target_id}")
        # Strip the term-frequency bag from the returned row — internal,
        # not part of the public dict shape.
        row = {k: v for k, v in row.items() if k != "_terms"}
        return {
            "kind": kind,
            "id": target_id,
            "updated": sorted(fields.keys()),
            "row": row,
        }
    finally:
        backend.close()


def cmd_delete(args: argparse.Namespace) -> dict:
    """Delete a Topic / Objective / Source / Thinking row.

    Spec L49 (CRUQD D) for plugin parity. Routes to the corresponding
    `store.delete_*` helper. Topic and Objective deletes cascade
    through their child Sources / Thinkings (matches the existing
    `store.delete_topic` / `delete_objective` semantics). Source and
    Thinking deletes are leaf-only.

    Payload shape:
      {"kind": "topic" | "objective" | "source" | "thinking", "id": int}

    Returns:
      {"kind": ..., "id": ..., "deleted": true} on success.
    Raises KeyError if the id does not exist (surfaced to caller as
    a structured `ok=false` JSON envelope by `main()`).
    """
    payload = json.loads(args.payload)
    kind = payload.get("kind")
    if kind not in ("topic", "objective", "source", "thinking"):
        raise ValueError(
            f"kind must be topic|objective|source|thinking, got {kind!r}"
        )
    if "id" not in payload:
        raise ValueError("payload missing required field 'id'")
    target_id = int(payload["id"])

    backend = get_default_backend(args.db)
    try:
        # Existence check before delete so the caller gets a clean error
        # instead of a silent no-op.
        if kind == "topic":
            existing = backend.get_node(target_id)
            if existing is None:
                raise KeyError(f"unknown topic: {target_id}")
            rg_store.delete_topic(backend, target_id)
        elif kind == "objective":
            if rg_store.get_objective(backend, target_id) is None:
                raise KeyError(f"unknown objective: {target_id}")
            rg_store.delete_objective(backend, target_id)
        elif kind == "source":
            if rg_store.get_source(backend, target_id) is None:
                raise KeyError(f"unknown source: {target_id}")
            rg_store.delete_source(backend, target_id)
        else:  # thinking
            if rg_store.get_thinking(backend, target_id) is None:
                raise KeyError(f"unknown thinking: {target_id}")
            backend.delete_node(target_id, cascade=False)
        return {"kind": kind, "id": target_id, "deleted": True}
    finally:
        backend.close()


def cmd_quality_trend(args: argparse.Namespace) -> dict:
    """Return the quality-over-time trend for an objective.

    Reads the JSONL run log at `<db_dir>/research_runs.jsonl` and
    returns a trend summary: list of (timestamp, quality, outcome)
    entries + monotonicity flag + improvement/regression counts.
    """
    from research_graph import evaluation as rg_eval

    objective_id = getattr(args, "objective_id", None)
    topic = getattr(args, "topic", None)
    question = getattr(args, "question", None)
    last_n = int(getattr(args, "last_n", None) or 20)

    if objective_id is None:
        if not topic or not question:
            raise ValueError(
                "quality-trend requires --objective-id OR both --topic and --question"
            )
        backend = get_default_backend(args.db)
        try:
            t = rg_store.get_topic_by_name(backend, topic)
            if t is None:
                return {"error": f"topic {topic!r} not found", "run_count": 0}
            for row in rg_store.query_objectives(backend, topic_id=int(t["id"])):
                if row.get("question") == question:
                    objective_id = int(row["id"])
                    break
            if objective_id is None:
                return {"error": f"objective not found for question {question!r}", "run_count": 0}
        finally:
            backend.close()

    return rg_eval.quality_trend(
        log_path=None, objective_id=int(objective_id),
        last_n=last_n, db_path=args.db,
    )


def cmd_daily_summary(args: argparse.Namespace) -> dict:
    """Run all objectives under a topic and return an aggregate daily summary."""
    from research_graph import schedule as rg_schedule
    return rg_schedule.run_daily_summary(
        args.db, args.topic,
        skip_external=bool(getattr(args, "skip_external", False)),
    )


def cmd_research_journey(args: argparse.Namespace) -> dict:
    """Continual-research entrypoint for the plugin (iter-5).

    Delegates to `research_graph.api.research_journey`, which:
      1. Resolves/creates (topic, objective) by exact-match name.
      2. Snapshots live-memory state BEFORE the run (live tip id +
         quality + snippet + sibling live thinkings).
      3. Runs one orchestrator cycle.
      4. Snapshots AFTER the run.
      5. Classifies outcome (cold_start | improved | branched |
         reused | rejected) and computes quality_delta.
      6. Returns a rich payload the plugin surfaces directly to
         OpenClaw without re-walking the graph.

    Payload shape matches `api.research_journey` return value; see
    its docstring for keys.
    """
    return rg_api.research_journey(
        args.db,
        topic=args.topic,
        question=args.question,
        skip_external=bool(args.skip_external),
        force_refresh=bool(args.force_refresh),
    )


def cmd_thinking_history(args: argparse.Namespace) -> dict:
    """Walk the supersession chain + sibling branches for an objective.

    Takes `objective_id` OR `(topic, question)` — if topic+question is
    supplied and the objective doesn't exist yet we return
    `{exists: false, ...}` rather than creating anything, because this
    command is read-only.
    """
    objective_id = getattr(args, "objective_id", None)
    if objective_id is None:
        topic_name = getattr(args, "topic", None)
        question = getattr(args, "question", None)
        if not topic_name or not question:
            raise ValueError(
                "thinking-history requires --objective-id OR "
                "both --topic and --question"
            )
        backend = get_default_backend(args.db)
        try:
            topic = rg_store.get_topic_by_name(backend, topic_name)
            if topic is None:
                return {
                    "objective_id": None,
                    "exists": False,
                    "live_tip_id": None,
                    "chain": [],
                    "siblings": [],
                    "warnings": [f"topic {topic_name!r} does not exist"],
                }
            for row in rg_store.query_objectives(backend, topic_id=int(topic["id"])):
                if row.get("question") == question:
                    objective_id = int(row["id"])
                    break
            if objective_id is None:
                return {
                    "objective_id": None,
                    "exists": False,
                    "live_tip_id": None,
                    "chain": [],
                    "siblings": [],
                    "warnings": [
                        f"objective with question {question!r} not found under topic "
                        f"{topic_name!r}"
                    ],
                }
        finally:
            backend.close()
    return rg_api.thinking_history(args.db, int(objective_id))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research-memory-plugin-helper")
    p.add_argument("--db", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--payload", required=True,
                        help="JSON string: {objective_id, sources:[{url,title,content}]}")

    query = sub.add_parser("query")
    query.add_argument("--payload", required=True,
                       help="JSON string: {question, kind?, top_k?, min_score?, topic_name?, topic_id?, objective_id?}")

    update = sub.add_parser("update")
    update.add_argument("--payload", required=True,
                        help="JSON string: {kind, id, fields:{...}}")

    delete = sub.add_parser("delete")
    delete.add_argument("--payload", required=True,
                        help="JSON string: {kind, id}")

    rt = sub.add_parser("research-topic")
    rt.add_argument("--topic", required=True)
    rt.add_argument("--question", required=True)
    rt.add_argument("--force-refresh", action="store_true")

    # iter-5: continual-research journey entrypoint. Returns memory-
    # before / run / memory-after / delta for plugin narration.
    journey = sub.add_parser("research-journey")
    journey.add_argument("--topic", required=True)
    journey.add_argument("--question", required=True)
    journey.add_argument("--skip-external", action="store_true",
                         help="Skip external fetch (offline/replay path).")
    journey.add_argument("--force-refresh", action="store_true",
                         help="Deprecated alias kept for CLI compat.")

    history = sub.add_parser("thinking-history")
    history.add_argument("--objective-id", type=int, default=None,
                         help="Objective id (preferred). If omitted you must pass --topic and --question.")
    history.add_argument("--topic", default=None)
    history.add_argument("--question", default=None)

    # iter-7: quality-over-time trend from the evaluation run log.
    qt = sub.add_parser("quality-trend")
    qt.add_argument("--objective-id", type=int, default=None)
    qt.add_argument("--topic", default=None)
    qt.add_argument("--question", default=None)
    qt.add_argument("--last-n", type=int, default=20)

    # iter-7: daily summary (run all objectives under a topic).
    ds = sub.add_parser("daily-summary")
    ds.add_argument("--topic", required=True)
    ds.add_argument("--skip-external", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "status":
            out: Any = cmd_status(args)
        elif args.cmd == "ingest":
            out = cmd_ingest(args)
        elif args.cmd == "query":
            out = cmd_query(args)
        elif args.cmd == "update":
            out = cmd_update(args)
        elif args.cmd == "delete":
            out = cmd_delete(args)
        elif args.cmd == "research-topic":
            out = cmd_research_topic(args)
        elif args.cmd == "research-journey":
            out = cmd_research_journey(args)
        elif args.cmd == "thinking-history":
            out = cmd_thinking_history(args)
        elif args.cmd == "quality-trend":
            out = cmd_quality_trend(args)
        elif args.cmd == "daily-summary":
            out = cmd_daily_summary(args)
        else:
            parser.print_help()
            return 1
    except Exception as exc:  # pragma: no cover - surfaced to caller
        print(json.dumps({
            "ok": False,
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }))
        # Always exit 0 so the JS caller does not collapse our structured
        # error JSON into a flat stderr string. Failure is signalled by
        # ok=false in the parsed object.
        return 0

    print(json.dumps(out, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
