# openclaw-research-graph

Continual research memory and retrieval layer for OpenClaw. Stdlib-only Python: SQLite for storage, hand-rolled TF-IDF for RAG, content-hash plus token-Jaccard dedup, and an actor/critic reasoning loop. Default external search is an offline JSON fixture so the system runs and tests deterministically without network.

## Goals (from spec.md)

- Long-running research over a topic with knowledge accumulation.
- CRUQD plus dedup plus RAG over the knowledge layer.
- Cold-start mode (external search only) and memory-augmented mode (retrieval plus optional refresh).
- OpenClaw-callable as a CLI subprocess and as a Python library.
- Daily 9 AM auto-research extensible structure.

## Install

No dependencies. Requires Python 3.9+ (uses `from __future__ import annotations`).

```
git clone <repo>
cd openclaw-research-graph
python3 -m research_graph init --db research.db
```

## CLI

```
python3 -m research_graph init [--db PATH]
python3 -m research_graph topic add <name>
python3 -m research_graph topic list [--json]
python3 -m research_graph objective add --topic <name> "<question>"
python3 -m research_graph objective list [--topic <name>] [--json]
python3 -m research_graph research <objective_id> [--force-refresh] [--json]
python3 -m research_graph query "<question>" [--kind sources|thinkings] [--top-k N] [--objective <id>] [--json]
python3 -m research_graph source list --objective <id> [--json]
python3 -m research_graph source delete <source_id>
python3 -m research_graph thinking list --objective <id> [--json]
python3 -m research_graph schedule cron --hour 9 --topic <name>
python3 -m research_graph schedule once --topic <name> [--json]
```

## OpenClaw integration

Three integration paths, in order of preference:

1. **Subprocess + JSON**

```
python3 -m research_graph research 17 --json
```

returns `{"objective": {...}, "mode": "memory_augmented", "new_sources": [...], "reused_sources": [...], "new_thinking": 42, "supersedes": 39, "rejected_reasons": []}`.

2. **Library call**

```python
from research_graph import api
result = api.research("research.db", objective_id=17)
```

3. **Daily 9 AM cron**

```
python3 -m research_graph schedule cron --hour 9 --topic "local-llm-bench"
# prints: 0 9 * * * /usr/bin/python3 -m research_graph schedule once --topic 'local-llm-bench' --db 'research.db'
```

`schedule once` is idempotent because dedup blocks identical sources from being inserted twice on the same day.

## Architecture

```
research_graph/
  storage.py       SQLite schema, CRUQD primitives, cascade delete
  dedup.py         Content-hash + token-Jaccard near-dup, separate spaces for source and thinking
  retrieval.py     TF-IDF index, filter then rank then relevance threshold
  search.py        Pluggable external search; default is offline JSON fixture
  reasoning.py     Actor synthesizes thinking from sources; critic enforces quote budget
  orchestrator.py  Cold-start vs memory-augmented research loop with supersession chain
  api.py           Library entry for OpenClaw
  cli.py           argparse subcommands with --json mode
  schedule.py      Cron line generator and idempotent runner
```

### Source vs thinking separation

Per spec: "source and thinking must not be mixed". Two enforcement points:

- `dedup` keeps independent dedup spaces -- a source and a thinking with identical content do not collide.
- `reasoning.critic_verify` rejects thinkings that quote source text verbatim beyond a small budget (max 2 quotes, max 200 chars per quote, no full-body repetition).

### Cold-start vs memory-augmented

`Orchestrator.research(objective_id, force_refresh=False)`:

- If the objective has zero sources: cold-start mode -- call external search, ingest hits.
- Otherwise: memory-augmented mode -- skip external search unless `force_refresh=True`. Retrieve top sources from local storage and feed them to the actor.
- When new evidence arrives during a refresh, the new thinking is created with `supersedes_id` pointing to the prior thinking. The prior is preserved as history.

### External search plug-in

Set `OPENCLAW_RESEARCH_SEARCH=mypkg.mymodule.factory` to swap in a real search backend. The factory must return an object with `search(query: str) -> list[SearchHit]`. The default `OfflineFixtureSearch` reads `research_graph/_fixtures/search.json` so tests and the cron job stay deterministic without network access.

## Tests

```
python3 -m unittest discover -s tests -v
```

23 tests, all passing under stdlib `unittest`. No third-party packages required.
