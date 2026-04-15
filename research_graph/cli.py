from __future__ import annotations

import argparse
import json
import sys

from . import api, retrieval, schedule, store
from .graph import get_default_backend
from .orchestrator import Orchestrator


DEFAULT_DB = "research_graph.json"


def _print(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, default=str, sort_keys=True))
    else:
        if isinstance(obj, list):
            for row in obj:
                print(row)
        else:
            print(obj)


def _strip_terms(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        c = dict(r)
        c.pop("_terms", None)
        c.pop("label", None)
        out.append(c)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research_graph")
    p.add_argument("--db", default=DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    topic = sub.add_parser("topic")
    topic_sub = topic.add_subparsers(dest="action", required=True)
    topic_add = topic_sub.add_parser("add")
    topic_add.add_argument("name")
    topic_list = topic_sub.add_parser("list")
    topic_list.add_argument("--json", dest="as_json", action="store_true")

    obj = sub.add_parser("objective")
    obj_sub = obj.add_subparsers(dest="action", required=True)
    obj_add = obj_sub.add_parser("add")
    obj_add.add_argument("--topic", required=True)
    obj_add.add_argument("question")
    obj_list = obj_sub.add_parser("list")
    obj_list.add_argument("--topic")
    obj_list.add_argument("--json", dest="as_json", action="store_true")

    research = sub.add_parser("research")
    research.add_argument("objective_id", type=int)
    research.add_argument("--force-refresh", action="store_true")
    research.add_argument("--json", dest="as_json", action="store_true")

    query = sub.add_parser("query")
    query.add_argument("question")
    query.add_argument("--kind", choices=["sources", "thinkings"], default="sources")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--objective", type=int)
    query.add_argument("--json", dest="as_json", action="store_true")

    src = sub.add_parser("source")
    src_sub = src.add_subparsers(dest="action", required=True)
    src_list = src_sub.add_parser("list")
    src_list.add_argument("--objective", type=int, required=True)
    src_list.add_argument("--json", dest="as_json", action="store_true")
    src_del = src_sub.add_parser("delete")
    src_del.add_argument("source_id", type=int)

    th = sub.add_parser("thinking")
    th_sub = th.add_subparsers(dest="action", required=True)
    th_list = th_sub.add_parser("list")
    th_list.add_argument("--objective", type=int, required=True)
    th_list.add_argument("--json", dest="as_json", action="store_true")

    walk = sub.add_parser("walk")
    walk_sub = walk.add_subparsers(dest="action", required=True)
    walk_4 = walk_sub.add_parser("4hop")
    walk_4.add_argument("--thinking", type=int, required=True)
    walk_4.add_argument("--json", dest="as_json", action="store_true")
    walk_co = walk_sub.add_parser("cocited")
    walk_co.add_argument("--thinking", type=int, required=True)
    walk_co.add_argument("--json", dest="as_json", action="store_true")
    walk_cross = walk_sub.add_parser("cross-topic-sources")
    walk_cross.add_argument("--json", dest="as_json", action="store_true")

    sch = sub.add_parser("schedule")
    sch_sub = sch.add_subparsers(dest="action", required=True)
    sch_cron = sch_sub.add_parser("cron")
    sch_cron.add_argument("--hour", type=int, default=9)
    sch_cron.add_argument("--topic", required=True)
    sch_once = sch_sub.add_parser("once")
    sch_once.add_argument("--topic", required=True)
    sch_once.add_argument("--json", dest="as_json", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db = args.db

    if args.cmd == "init":
        backend = get_default_backend(db)
        try:
            pass
        finally:
            backend.close()
        print(f"initialized {db}")
        return 0

    if args.cmd == "topic":
        backend = get_default_backend(db)
        try:
            if args.action == "add":
                tid = store.create_topic(backend, args.name)
                print(tid)
            elif args.action == "list":
                _print(_strip_terms(store.list_topics(backend)), getattr(args, "as_json", False))
        finally:
            backend.close()
        return 0

    if args.cmd == "objective":
        backend = get_default_backend(db)
        try:
            if args.action == "add":
                topic = store.get_topic_by_name(backend, args.topic)
                if topic is None:
                    print(f"unknown topic: {args.topic}", file=sys.stderr)
                    return 2
                oid = store.create_objective(backend, int(topic["id"]), args.question)
                print(oid)
            elif args.action == "list":
                topic_id = None
                if args.topic:
                    t = store.get_topic_by_name(backend, args.topic)
                    if t is None:
                        print(f"unknown topic: {args.topic}", file=sys.stderr)
                        return 2
                    topic_id = int(t["id"])
                _print(
                    _strip_terms(store.query_objectives(backend, topic_id=topic_id)),
                    getattr(args, "as_json", False),
                )
        finally:
            backend.close()
        return 0

    if args.cmd == "research":
        result = api.research(db, args.objective_id, force_refresh=args.force_refresh)
        _print(result, args.as_json)
        return 0

    if args.cmd == "query":
        backend = get_default_backend(db)
        try:
            if args.kind == "sources":
                rows = retrieval.search_sources(
                    backend, args.question, objective_id=args.objective, top_k=args.top_k
                )
            else:
                rows = retrieval.search_thinkings(
                    backend, args.question, objective_id=args.objective, top_k=args.top_k
                )
            _print(_strip_terms(rows), args.as_json)
        finally:
            backend.close()
        return 0

    if args.cmd == "source":
        backend = get_default_backend(db)
        try:
            if args.action == "list":
                _print(_strip_terms(store.list_sources(backend, args.objective)),
                       getattr(args, "as_json", False))
            elif args.action == "delete":
                store.delete_source(backend, args.source_id)
                print(f"deleted {args.source_id}")
        finally:
            backend.close()
        return 0

    if args.cmd == "thinking":
        backend = get_default_backend(db)
        try:
            if args.action == "list":
                _print(
                    _strip_terms(store.list_thinkings(backend, args.objective)),
                    getattr(args, "as_json", False),
                )
        finally:
            backend.close()
        return 0

    if args.cmd == "walk":
        backend = get_default_backend(db)
        try:
            if args.action == "4hop":
                result = store.four_hop_evidence_walk(backend, args.thinking)
                result = {
                    "start": {k: v for k, v in result["start"].items() if k != "_terms"},
                    "shared_sources": _strip_terms(result["shared_sources"]),
                    "related_thinkings": _strip_terms(result["related_thinkings"]),
                    "new_sources": _strip_terms(result["new_sources"]),
                }
                _print(result, getattr(args, "as_json", False))
            elif args.action == "cocited":
                rows = store.cocited_thinkings(backend, args.thinking)
                _print(_strip_terms(rows), getattr(args, "as_json", False))
            elif args.action == "cross-topic-sources":
                rows = store.cross_topic_shared_sources(backend)
                for r in rows:
                    r["sources"] = _strip_terms(r["sources"])
                _print(rows, getattr(args, "as_json", False))
        finally:
            backend.close()
        return 0

    if args.cmd == "schedule":
        if args.action == "cron":
            print(schedule.cron_line(args.hour, args.topic, db))
        elif args.action == "once":
            results = schedule.run_once_for_topic(db, args.topic)
            _print(results, getattr(args, "as_json", False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
