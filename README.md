# openclaw-research-graph

Continual research system with a Neo4j-backed knowledge graph for OpenClaw.

This system enables OpenClaw to perform iterative research on specific topics over time, continuously incorporating new external information, updating existing conclusions, and reusing accumulated knowledge. It is designed as a long-lived research memory and retrieval engine, not a one-shot summarizer.

## Architecture

The knowledge graph is organized as a hierarchy of nodes in Neo4j:

- **Topic** -- broad, long-lived research area
- **ResearchObjective** -- concrete research goal within a topic
- **SourceSummary** -- summarized external source (web page, paper, video)
- **SourceChunk** -- text chunk belonging to a source summary
- **ThinkingSummary** -- analytical reflection or synthesis node
- **ThinkingChunk** -- text chunk belonging to a thinking summary

Relationships: `BELONGS_TO` (parent association), `CHILD_OF` (chunk to summary), `REFERENCES` (cross-references between summaries).

All nodes with text content carry vector embeddings for semantic search via Neo4j vector indexes.

## Quick start

### Install

```bash
pip install -e ".[dev]"
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | (empty) | Neo4j password |
| `OLLAMA_URL` | `http://localhost:11434/api/embed` | Ollama embedding endpoint |
| `RESEARCH_GRAPH_EMBED_MODEL` | `qwen3-embedding:0.6b` | Embedding model name |

### Initialize the database

Run the Cypher scripts in `schema/` against your Neo4j instance:

1. `schema/init.cypher` -- constraints and indexes
2. `schema/vector-indexes.cypher` -- vector indexes for semantic search

### CLI usage

```bash
research-graph find-or-create-objective --objective-text "Best local LLM for 16GB Mac Mini"
research-graph get-context --objective-id <id>
research-graph search --query "local LLM performance"
research-graph get-node-bundle --node-id <id>
research-graph review-objective-state --objective-id <id>
research-graph upsert-sources --topic-id <id> --input sources.json
research-graph upsert-thinkings --topic-id <id> --input thinkings.json
```

## Core actions

| Action | Description |
|---|---|
| `find_or_create_research_objective` | Find an existing objective by text similarity or create a new one with its topic |
| `get_research_context` | Retrieve the full context for an objective or topic (thinking summaries, source summaries, chunks) |
| `upsert_source_batch` | Create or deduplicate source summaries with chunking and embedding |
| `upsert_thinking_batch` | Create thinking summaries with chunking, embedding, and cross-references |
| `search_graph` | Semantic vector search across all node types with deduplication and ranking |
| `get_node_bundle` | Deep inspection of a single node with its parents, children, and references |
| `review_objective_state` | Comprehensive review of an objective combining context, search, and open questions |

## Schema reference

See `schema/schema.md` for the full node label and relationship specification, time format, embedding policy, and parent rules.

## Tests

```bash
python -m pytest tests/ -v
```

Tests for `utils`, `chunking`, and `embed` modules run without a Neo4j instance.
