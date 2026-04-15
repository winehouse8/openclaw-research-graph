# research-memory (OpenClaw plugin)

`research-memory` is a native OpenClaw plugin that wraps the
`openclaw-research-graph` Python backend and positions it as the canonical
long-term memory + research orchestration layer for an OpenClaw instance.
Claude Code and Codex harnesses (and any other executor plugins) sit *below*
it as low-level executors — they are called only when a research step needs
deep code analysis, long-form reasoning, or multi-file implementation, and
they read/write memory through this plugin's tool surface instead of owning
their own private state.

The plugin intentionally does NOT spawn harness subagents itself. It exposes
the MEMORY layer that `sdk-agent-harness` and `codex-harness` plugins — and
any direct LLM caller — can consume via five typed tools and four slash
commands.

## Files

```
plugins/research-memory/
  README.md              # this file
  openclaw.plugin.json   # native plugin manifest (id, config schema, tools)
  package.json           # npm metadata + openclaw.extensions entrypoint
  index.js               # definePluginEntry(...) entrypoint (tools + commands)
  plugin_helper.py       # stdlib Python bridge for status/ingest/research-topic
  test.js                # node --test suite that drives the plugin handlers end to end
```

## Delegation boundary

**OpenClaw owns (via this plugin):**

- Research objective lifecycle (create, resume, force-refresh)
- Long-term memory canonical store — the research graph with its
  topic/objective/source/thinking nodes and `HAS_*`, `CITES`, `SUPERSEDES`,
  `REUSES` edges
- Scheduling — the `/research` command and `schedule once` CLI path work
  without any live agent attached, so cron re-research is a first-class
  operation
- Cross-session retrieval — `memory_search` and `graph_walk` are callable by
  any agent, channel, or human from any OpenClaw session
- Source / thinking purity and provenance invariants — those are enforced at
  the graph layer before any executor sees them

**Claude / Codex harnesses own:**

- Low-level reasoning on material the memory layer has already grounded
- Multi-file code implementation and refactors
- Long-form deep-analysis passes over specific sources the memory layer
  retrieved for them

Harness calls go through OpenClaw's existing `sdk-agent-harness` /
`codex-harness` plugins. The `research-memory` plugin deliberately does NOT
call `runEmbeddedAgent`, `api.runtime.subagent`, or any harness surface —
it is purely the memory + orchestration layer those harnesses sit below.

**Why OpenClaw-owned is correct:**

- Memory must outlive any single Claude/Codex session. A Claude worker that
  exits drops its context. The graph DB persists on disk between sessions
  and across worker restarts.
- Scheduled (cron) re-research must run without a live agent attached.
  OpenClaw's scheduler can invoke `research_topic` on its own.
- Multiple different agents/channels (actor-cc-bot, critic-cc-bot,
  codex-worker, a human `/memory-search`) must call the same memory. If
  Claude/Codex owned it, each session would need to re-boot the graph
  context.
- Source/thinking purity + provenance invariants are enforced at a layer
  above any individual executor. The `REL_SUPERSEDES` / `REL_REUSES`
  invariants and 4hop walk semantics are not something an ad-hoc session
  context can preserve.

For the architecture rationale behind the graph backend itself (SQLite vs
Neo4j, why a graph, how the backends are pluggable) see
[`docs/sqlite-vs-neo4j-analysis.md`](../../docs/sqlite-vs-neo4j-analysis.md)
in the repo root.

## Tools

All five tools are fully implemented and end-to-end tested against the real
Python graph backend. See `test.js`.

| Tool             | Input                                                                        | Returns                                                                                     | Status       |
| ---------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------ |
| `research_topic` | `{topic, question, force_refresh?}`                                          | `{topic, topic_id, question, objective_id, research: ResearchResult}`                       | full e2e     |
| `memory_search`  | `{query, kind: sources\|thinkings, top_k?, objective_id?}`                   | `{kind, query, topK, hits: [...]}`                                                          | full e2e     |
| `memory_ingest`  | `{objective_id, sources: [{url, title, content}]}`                           | `{objective_id, inserted: [int], skipped: [{url, reason, source_id?}]}`                     | full e2e     |
| `graph_walk`     | `{mode: 4hop\|cocited\|cross-topic-sources, anchor_id?}`                     | `{mode, anchorId, result}` (shape depends on mode, see `research_graph/store.py`)           | full e2e     |
| `memory_status`  | `{}`                                                                         | `{db, counts: {topics, objectives, sources, thinkings, supersessions, reuses}}`             | full e2e     |

### Deviation note: `supersession-chain`

The original task spec mentions a `supersession-chain` walk mode. The current
`openclaw-research-graph` CLI only exposes `4hop`, `cocited`, and
`cross-topic-sources` as first-class walk subcommands — supersession chains
are reachable via the `SUPERSEDES` edge through `4hop`'s `related_thinkings`
section, but there is no dedicated `walk supersession-chain` subparser. The
plugin:

1. Accepts only the three modes the Python CLI actually exposes.
2. Throws a structured error (with a clear message and the list of supported
   modes) if a caller asks for `supersession-chain`.
3. Documents this here. Adding a dedicated supersession-chain walker is a
   follow-up PR against `research_graph/store.py` + `cli.py`, not something
   the plugin should fake on its own.

The plugin's `graph_walk` tool rejects the unsupported mode explicitly
(verified by `test.js`).

## Slash commands

| Command                     | Usage                                                            | Maps to                            |
| --------------------------- | ---------------------------------------------------------------- | ---------------------------------- |
| `/research`                 | `/research <topic>::<question>`                                  | `research_topic` tool              |
| `/memory-search`            | `/memory-search <sources\|thinkings>::<query>`                   | `memory_search` tool               |
| `/memory-status`            | `/memory-status`                                                 | `memory_status` tool               |
| `/graph-walk`               | `/graph-walk <4hop\|cocited\|cross-topic-sources> [anchor_id]`   | `graph_walk` tool                  |

The `::` separator keeps multi-word topics + questions legible without
needing quoting tricks from chat channels.

## Configuration

Config lives under `plugins.entries.research-memory.config` in the OpenClaw
config file. All three fields are optional.

```json5
{
  plugins: {
    entries: {
      "research-memory": {
        enabled: true,
        config: {
          // Filesystem path to the research_graph JSON/Neo4j backing store.
          // Falls back to the RESEARCH_GRAPH_DB env var, then to
          // ./research_graph.json relative to the gateway's cwd.
          dbPath: "/var/lib/openclaw/research_graph.json",

          // Python interpreter to use for the subprocess bridge.
          // Defaults to python3.
          python: "python3",

          // Override for the research_graph Python package parent dir.
          // Falls back to RESEARCH_GRAPH_PACKAGE_DIR env var or to the
          // plugin's own repo root (../..  from the plugin dir).
          packageDir: "/opt/openclaw-research-graph"
        }
      }
    }
  }
}
```

### Env vars (lowest precedence after inline config)

- `RESEARCH_GRAPH_DB` — default db path for the research graph
- `RESEARCH_GRAPH_PACKAGE_DIR` — default python package parent dir
- `PYTHON` — override the python interpreter

### Neo4j backend

The Python package already supports switching from the in-memory JSON
backend to Neo4j. Point `RESEARCH_GRAPH_DB` (and any other
research-graph-specific env vars documented in `research_graph/graph/`) at
your Neo4j instance. The plugin is backend-agnostic because it never talks
to the backend directly — it only talks to `python3 -m research_graph` and
`plugin_helper.py`.

## Install and run locally

The plugin is a local-path install. From the OpenClaw host:

```bash
# Install the plugin from a local path (links into plugins.load.paths).
openclaw plugins install -l /path/to/openclaw-research-graph/plugins/research-memory

# Alternatively, install by copying the directory into the plugins root:
openclaw plugins install /path/to/openclaw-research-graph/plugins/research-memory

# Verify it loaded:
openclaw plugins list --verbose
openclaw plugins inspect research-memory

# Restart the gateway for config changes to take effect:
openclaw gateway restart
```

To enable the plugin (and/or its tools) explicitly in config:

```json5
{
  plugins: {
    allow: ["research-memory"],
    entries: {
      "research-memory": { enabled: true }
    }
  },
  tools: {
    allow: ["research_topic", "memory_search", "memory_ingest", "graph_walk", "memory_status"]
  }
}
```

## Tests

```bash
cd plugins/research-memory
node --test test.js
```

The test suite:

1. Spins up a tempdir-backed `research_graph.json` DB per test.
2. Calls `handlers.researchTopic(...)` with a canned `(topic, question)` pair
   and verifies the ResearchResult shape + the resulting topic/objective ids.
3. Calls `handlers.memoryStatus(...)` before and after and asserts counts
   increased.
4. Calls `handlers.memoryIngest(...)` to add a deterministic source, then
   calls `handlers.memorySearch(...)` for a known-needle term and asserts
   there is at least one hit.
5. Calls `handlers.graphWalk(... mode: "4hop" ...)` with a real thinking
   anchor id and asserts the 4hop result shape
   (`start`/`shared_sources`/`related_thinkings`/`new_sources`).
6. Calls `handlers.graphWalk(... mode: "supersession-chain" ...)` and asserts
   the plugin rejects it with a clear error (the deviation note above).

The suite does NOT require a running OpenClaw daemon — `index.js` exports
`handlers` directly and the default entry's SDK import is lazy, so the file
can be imported and driven from plain `node --test`.

## Assumptions (documented because the docs did not directly answer)

- **Entry file extension.** The SDK docs reference `./index.ts` in examples,
  but every *installed* bundled plugin on a production OpenClaw host is
  compiled to `./index.js` (the `duckduckgo`, `zalouser`, `together`, etc.
  extensions under `dist/extensions/`). This plugin ships `./index.js`
  directly to match the runtime loader's actual input.
- **Registering commands.** The SDK Overview lists
  `api.registerCommand(def)` as a registration method but does not include
  the full `def` shape in the reference. The plugin uses a conservative
  `{name, description, async execute({argv})}` shape. If `registerCommand`
  is unavailable on a given host (older SDK), the plugin uses optional
  chaining (`api.registerCommand?.(...)`) so registration silently skips
  instead of crashing the plugin load.
- **Typebox availability.** The quickstart uses `Type.Object(...)` from
  `@sinclair/typebox`. The plugin tries to import it and falls back to a
  plain JSON-Schema-shaped `parameters` object if the import fails, so the
  plugin still loads even if typebox is not reachable in the host's resolver
  path.
- **Walk mode coverage.** `supersession-chain` is not exposed by the current
  Python CLI; see the deviation note above.
