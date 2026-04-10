import argparse
import json
from pathlib import Path

from .actions import (
    find_or_create_research_objective,
    get_node_bundle,
    get_research_context,
    review_objective_state,
    search_graph,
    upsert_source_batch,
    upsert_thinking_batch,
)


def _print(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(prog='research-graph')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('find-or-create-objective')
    p.add_argument('--objective-text', required=True)
    p.add_argument('--topic-hint')
    p.add_argument('--name-hint')
    p.add_argument('--success-criteria')

    p = sub.add_parser('get-context')
    p.add_argument('--objective-id')
    p.add_argument('--topic-id')
    p.add_argument('--thinking-limit', type=int, default=5)
    p.add_argument('--source-limit', type=int, default=5)
    p.add_argument('--chunk-limit', type=int, default=6)
    p.add_argument('--include-embedding', action='store_true')

    p = sub.add_parser('search')
    p.add_argument('--query', required=True)
    p.add_argument('--topic-id')
    p.add_argument('--objective-id')
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--include-embedding', action='store_true')

    p = sub.add_parser('get-node-bundle')
    p.add_argument('--node-id', required=True)
    p.add_argument('--include-embedding', action='store_true')

    p = sub.add_parser('review-objective-state')
    p.add_argument('--objective-id', required=True)
    p.add_argument('--query')
    p.add_argument('--thinking-limit', type=int, default=5)
    p.add_argument('--source-limit', type=int, default=5)
    p.add_argument('--chunk-limit', type=int, default=6)
    p.add_argument('--search-limit', type=int, default=6)

    p = sub.add_parser('upsert-sources')
    p.add_argument('--topic-id', required=True)
    p.add_argument('--objective-id')
    p.add_argument('--input', required=True)

    p = sub.add_parser('upsert-thinkings')
    p.add_argument('--topic-id', required=True)
    p.add_argument('--objective-id')
    p.add_argument('--input', required=True)

    args = parser.parse_args()

    if args.command == 'find-or-create-objective':
        _print(find_or_create_research_objective(args.objective_text, args.topic_hint, args.name_hint, args.success_criteria))
    elif args.command == 'get-context':
        _print(get_research_context(args.objective_id, args.topic_id, args.thinking_limit, args.source_limit, args.chunk_limit, args.include_embedding))
    elif args.command == 'search':
        _print(search_graph(args.query, args.topic_id, args.objective_id, args.limit, args.include_embedding))
    elif args.command == 'get-node-bundle':
        _print(get_node_bundle(args.node_id, args.include_embedding))
    elif args.command == 'review-objective-state':
        _print(review_objective_state(args.objective_id, args.query, args.thinking_limit, args.source_limit, args.chunk_limit, args.search_limit))
    elif args.command == 'upsert-sources':
        data = json.loads(Path(args.input).read_text())
        _print(upsert_source_batch(args.topic_id, data, args.objective_id))
    elif args.command == 'upsert-thinkings':
        data = json.loads(Path(args.input).read_text())
        _print(upsert_thinking_batch(args.topic_id, data, args.objective_id))


if __name__ == '__main__':
    main()
