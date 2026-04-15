from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import api, retrieval, schedule, storage
from .orchestrator import Orchestrator


DEFAULT_DB = "research_graph.db"


def _print(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, default=str, sort_keys=True))
    else:
        if isinstance(obj, list):
            for row in obj:
                print(row)
        else:
            print(obj)


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
        conn = storage.connect(db)
        conn.close()
        print(f"initialized {db}")
        return 0

    if args.cmd == "topic":
        conn = storage.connect(db)
        try:
            if args.action == "add":
                tid = storage.create_topic(conn, args.name)
                print(tid)
            elif args.action == "list":
                _print(storage.list_topics(conn), getattr(args, "as_json", False))
        finally:
            conn.close()
        return 0

    if args.cmd == "objective":
        conn = storage.connect(db)
        try:
            if args.action == "add":
                topic = storage.get_topic_by_name(conn, args.topic)
                if topic is None:
                    print(f"unknown topic: {args.topic}", file=sys.stderr)
                    return 2
                oid = storage.create_objective(conn, int(topic["id"]), args.question)
                print(oid)
            elif args.action == "list":
                topic_id = None
                if args.topic:
                    t = storage.get_topic_by_name(conn, args.topic)
                    if t is None:
                        print(f"unknown topic: {args.topic}", file=sys.stderr)
                        return 2
                    topic_id = int(t["id"])
                _print(
                    storage.query_objectives(conn, topic_id=topic_id),
                    getattr(args, "as_json", False),
                )
        finally:
            conn.close()
        return 0

    if args.cmd == "research":
        result = api.research(db, args.objective_id, force_refresh=args.force_refresh)
        _print(result, args.as_json)
        return 0

    if args.cmd == "query":
        conn = storage.connect(db)
        try:
            if args.kind == "sources":
                rows = retrieval.search_sources(
                    conn, args.question, objective_id=args.objective, top_k=args.top_k
                )
            else:
                rows = retrieval.search_thinkings(
                    conn, args.question, objective_id=args.objective, top_k=args.top_k
                )
            _print(rows, args.as_json)
        finally:
            conn.close()
        return 0

    if args.cmd == "source":
        conn = storage.connect(db)
        try:
            if args.action == "list":
                _print(storage.list_sources(conn, args.objective), getattr(args, "as_json", False))
            elif args.action == "delete":
                storage.delete_source(conn, args.source_id)
                print(f"deleted {args.source_id}")
        finally:
            conn.close()
        return 0

    if args.cmd == "thinking":
        conn = storage.connect(db)
        try:
            if args.action == "list":
                _print(
                    storage.list_thinkings(conn, args.objective),
                    getattr(args, "as_json", False),
                )
        finally:
            conn.close()
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
