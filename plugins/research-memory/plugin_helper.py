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
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            existing = rg_store.find_source_by_hash_in_objective(
                backend, objective_id, content_hash
            )
            if existing is not None:
                skipped.append({"url": url, "reason": "duplicate",
                                "source_id": int(existing["id"])})
                continue
            sid = rg_store.insert_source(
                backend,
                objective_id=objective_id,
                url=url,
                title=title,
                content=content,
                content_hash=content_hash,
                search_query=None,
            )
            # Ingested sources must also be indexed in the term-frequency
            # retrieval layer, otherwise memory_search cannot find them.
            rg_retrieval.index_source(backend, int(sid), content)
            inserted.append(int(sid))
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research-memory-plugin-helper")
    p.add_argument("--db", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--payload", required=True,
                        help="JSON string: {objective_id, sources:[{url,title,content}]}")

    rt = sub.add_parser("research-topic")
    rt.add_argument("--topic", required=True)
    rt.add_argument("--question", required=True)
    rt.add_argument("--force-refresh", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "status":
            out: Any = cmd_status(args)
        elif args.cmd == "ingest":
            out = cmd_ingest(args)
        elif args.cmd == "research-topic":
            out = cmd_research_topic(args)
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
