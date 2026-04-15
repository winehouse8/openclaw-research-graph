# openclaw-research-graph

Continual research memory and retrieval layer for OpenClaw. The storage layer is a graph: labelled nodes (Topic, Objective, Source, Thinking) connected by typed relationships (HAS_OBJECTIVE, HAS_SOURCE, HAS_THINKING, CITES, SUPERSEDES, REUSES). Two interchangeable backends speak the same protocol: a pure-stdlib `InMemoryGraphBackend` used by tests and the default CLI, and a `Neo4jBackend` wrapping the official driver. Flip between them with a single environment variable.

## Install

No dependencies required for the default (in-memory) backend. Python 3.9+.

```
git clone <repo>
cd openclaw-research-graph
python3 -m research_graph init --db research.json
```

To use the Neo4j backend: install the `neo4j` driver (already present on the dev host) and set the three env vars described below.

## Architecture

```
research_graph/
  graph/
    model.py          Node, Edge, NodeRef dataclasses + label/rel constants
    backend.py        GraphBackend Protocol (create_node, create_edge, ...)
    inmemory.py       InMemoryGraphBackend (adjacency-list + JSON persistence)
    neo4j_backend.py  Neo4jBackend (bolt driver, Cypher translation)
    factory.py        get_default_backend() - env-driven selection
  store.py            Domain layer: create_topic, create_objective, insert_source,
                      insert_thinking, supersede, reuse, four_hop_evidence_walk,
                      cocited_thinkings, cross_topic_shared_sources, ...
  dedup.py            Content-hash + token-Jaccard near-dup (graph-aware)
  retrieval.py        TF-IDF over node properties (backend-agnostic)
  reasoning.py        actor_propose + critic_verify (reads via store)
  orchestrator.py     Cold-start vs memory-augmented loop; persists REUSES edges
  api.py              Library entry point
  cli.py              argparse subcommands, including walk 4hop / cocited / cross-topic
  schedule.py         Cron line generator + idempotent runner
```

## Graph schema

Node labels:

- `Topic { id, name, created_at }`
- `Objective { id, question, status, created_at, updated_at, topic_id }`
- `Source { id, url, title, content, content_hash, fetched_at, search_query, objective_id }`
- `Thinking { id, content, content_hash, author, created_at, supersedes_id, objective_id }`

Relationships:

- `(Topic)-[:HAS_OBJECTIVE]->(Objective)`
- `(Objective)-[:HAS_SOURCE]->(Source)`
- `(Objective)-[:HAS_THINKING]->(Thinking)`
- `(Thinking)-[:CITES]->(Source)` — replaces the legacy `supports_source_ids` blob
- `(Thinking)-[:SUPERSEDES]->(Thinking)` — replaces `supersedes_id` self-FK
- `(Thinking)-[:REUSES]->(Thinking)` — new: dedup-collapse fact that used to be in-memory only

Uniqueness constraints (enforced by Cypher on Neo4j, by the store layer on in-memory): `Topic.name` unique, `Source.content_hash` unique within an objective, `Thinking.content_hash` unique within an objective.

## Backends

Selection is driven by `OPENCLAW_GRAPH_BACKEND`:

```
# default: stdlib in-memory graph persisted to JSON at --db PATH
python3 -m research_graph --db research.json init

# real Neo4j backend
export OPENCLAW_GRAPH_BACKEND=neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...
export NEO4J_DATABASE=neo4j     # optional
python3 -m research_graph init
```

## CLI

```
python3 -m research_graph init [--db PATH]
python3 -m research_graph topic add <name>
python3 -m research_graph topic list [--json]
python3 -m research_graph objective add --topic <name> "<question>"
python3 -m research_graph objective list [--topic <name>] [--json]
python3 -m research_graph research <objective_id> [--force-refresh] [--json]
python3 -m research_graph query "<q>" [--kind sources|thinkings] [--objective <id>] [--json]
python3 -m research_graph source list --objective <id> [--json]
python3 -m research_graph thinking list --objective <id> [--json]
python3 -m research_graph walk 4hop --thinking <id> [--json]
python3 -m research_graph walk cocited --thinking <id> [--json]
python3 -m research_graph walk cross-topic-sources [--json]
python3 -m research_graph schedule cron --hour 9 --topic <name>
python3 -m research_graph schedule once --topic <name> [--json]
```

## Multi-hop queries

`store.four_hop_evidence_walk(thinking_id)` is the canonical example from `docs/sqlite-vs-neo4j-analysis.md`. Against Neo4j it translates to:

```cypher
MATCH (t0:Thinking {id: $tid})-[:CITES]->(s0:Source)<-[:CITES]-(t1:Thinking)
WHERE t1.id <> t0.id
OPTIONAL MATCH (t1)-[:CITES]->(s1:Source)
WHERE NOT (t0)-[:CITES]->(s1)
RETURN t0,
       collect(DISTINCT s0) AS shared_sources,
       collect(DISTINCT t1) AS related_thinkings,
       collect(DISTINCT s1) AS new_sources
```

`store.cocited_thinkings` (Cypher equivalent in `store.py`), `store.supersession_chain`, and `store.cross_topic_shared_sources` follow the same pattern: pure Python adjacency walks for the in-memory backend, equivalent Cypher on Neo4j.

## Tests

```
python3 -m unittest discover -s tests -v
```

50 tests. The Neo4j smoke test (`tests/test_neo4j_backend_smoke.py`) skips unless `OC_TEST_NEO4J_URI`, `OC_TEST_NEO4J_USER`, `OC_TEST_NEO4J_PASSWORD` are set, so the default suite runs without touching any live database.

## Demos

```
python3 scripts/demo_research_graph.py     # in-memory; 3 topics, 5 days, supersession + 4-hop walk
python3 scripts/demo_neo4j_backend.py      # real Neo4j; skips cleanly if NEO4J_* unset
```

## Design notes

See `docs/sqlite-vs-neo4j-analysis.md` for the rationale behind the graph-first model and the motivating multi-hop queries.
