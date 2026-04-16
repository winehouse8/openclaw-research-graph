// OpenClaw research-memory plugin entry.
//
// This plugin positions the openclaw-research-graph Python backend as the
// canonical long-term memory + research orchestration layer for OpenClaw,
// and exposes it to agents (Claude/Codex harnesses included) through five
// typed tools and four slash commands. See ./README.md for the delegation
// boundary rationale.
//
// The JS entry intentionally stays thin: each tool shells out to either
// `python3 -m research_graph ...` (for operations the CLI already exposes
// in JSON) or to `plugin_helper.py` (for status/ingest/research-topic that
// need direct Python-package access). Both code paths reuse the same
// `research_graph` Python package, so there is exactly one canonical graph
// store.
//
// Tests (test.js) import the `handlers` object below directly and drive the
// bridge without booting an OpenClaw runtime or constructing a fake `api`.
// This is the "plugin entry should be importable and invokable directly for
// testing" path the spec asks for.

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import process from "node:process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_DB = process.env.RESEARCH_GRAPH_DB || "research_graph.json";

/**
 * Resolve the effective runtime settings for this plugin invocation.
 * Precedence for every field is: caller-supplied > pluginConfig > env var >
 * built-in default.
 */
export function resolveSettings(pluginConfig = {}, overrides = {}) {
  const dbPath =
    overrides.dbPath ||
    pluginConfig.dbPath ||
    process.env.RESEARCH_GRAPH_DB ||
    DEFAULT_DB;
  const python =
    overrides.python || pluginConfig.python || process.env.PYTHON || "python3";
  const packageDir =
    overrides.packageDir ||
    pluginConfig.packageDir ||
    process.env.RESEARCH_GRAPH_PACKAGE_DIR ||
    path.resolve(__dirname, "..", "..");
  return { dbPath, python, packageDir };
}

function runPython(python, args, { packageDir, timeoutMs = 120_000 } = {}) {
  return new Promise((resolve, reject) => {
    const env = { ...process.env };
    if (packageDir) {
      env.RESEARCH_GRAPH_PACKAGE_DIR = packageDir;
      env.PYTHONPATH = packageDir +
        (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : "");
    }
    const child = spawn(python, args, {
      cwd: packageDir || process.cwd(),
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    const to = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(
        `python bridge timed out after ${timeoutMs}ms: ${python} ${args.join(" ")}`,
      ));
    }, timeoutMs);
    child.on("error", (err) => {
      clearTimeout(to);
      reject(err);
    });
    child.on("close", (code) => {
      clearTimeout(to);
      // Always resolve with the captured streams + exit code. The caller
      // (each handler) is responsible for parsing stdout JSON and surfacing
      // structured errors via throwStructured. This lets us treat both
      // exit-0 in-band errors (plugin_helper) and non-zero CLI errors
      // (`python -m research_graph`) uniformly: parse stdout first, fall
      // back to stderr only if stdout has no parseable JSON.
      resolve({ stdout, stderr, code });
    });
  });
}

function tryParseJson(stdout) {
  const trimmed = stdout.trim();
  if (!trimmed) return null;
  const lines = trimmed.split(/\n/).filter((l) => l.trim().length > 0);
  const last = lines[lines.length - 1];
  try {
    return JSON.parse(last);
  } catch {
    return null;
  }
}

function parseJson(stdout) {
  const trimmed = stdout.trim();
  if (!trimmed) return null;
  // The CLI sometimes prints multiple JSON lines (e.g. `topic add` prints an
  // int id). We take the last non-empty line as the JSON payload.
  const lines = trimmed.split(/\n/).filter((l) => l.trim().length > 0);
  const last = lines[lines.length - 1];
  try {
    return JSON.parse(last);
  } catch (err) {
    throw new Error(`failed to parse JSON from python bridge: ${err.message}\nraw: ${trimmed}`);
  }
}

/**
 * Throw a structured Error built from a parsed JSON payload that signals
 * failure (either `ok === false` or a top-level `error` key). The original
 * parsed object is attached as `.cause` so callers can branch on
 * `err.cause.type` programmatically. The thrown message includes `type`
 * for human readability.
 */
function throwStructured(parsed, fallback) {
  const type = parsed && typeof parsed.type === "string" ? parsed.type : "Error";
  const message = parsed && typeof parsed.error === "string"
    ? parsed.error
    : (fallback || "python bridge reported failure");
  const err = new Error(`python bridge error [${type}]: ${message}`);
  err.cause = parsed;
  throw err;
}

/**
 * Run a python subprocess and parse its stdout JSON, throwing a structured
 * error if the helper signalled failure (in-band on exit 0) or if the CLI
 * exited non-zero. Returns the parsed JSON payload on success.
 */
async function runPythonJson(python, args, opts = {}) {
  const { stdout, stderr, code } = await runPython(python, args, opts);
  const parsed = tryParseJson(stdout);
  if (code === 0) {
    if (parsed && parsed.ok === false) {
      // In-band failure from plugin_helper.py (always exits 0, signals via ok)
      throwStructured(parsed, stderr.trim());
    }
    if (parsed === null) {
      // Empty stdout on success -> let parseJson surface a clean error
      return parseJson(stdout);
    }
    return parsed;
  }
  // Non-zero exit: prefer parseable structured JSON, else fall back to stderr.
  if (parsed && (parsed.ok === false || parsed.error)) {
    throwStructured(parsed, stderr.trim());
  }
  throw new Error(
    `python bridge exited ${code}: ${stderr.trim() || stdout.trim()}`,
  );
}

// ---------------------------------------------------------------------------
// Preflight: verify the research_graph python package is importable from the
// configured packageDir using the same spawn/cwd/env the real tool calls use.
// Cached on the settings object so it runs at most once per process per
// settings instance.

async function preflightCheck(settings) {
  if (settings.__preflightOk) return;
  if (settings.__preflightPromise) {
    await settings.__preflightPromise;
    return;
  }
  settings.__preflightPromise = (async () => {
    let stdout = "", stderr = "", code = 0;
    try {
      const r = await runPython(
        settings.python,
        ["-c", "import research_graph, sys; print(research_graph.__file__)"],
        { packageDir: settings.packageDir, timeoutMs: 30_000 },
      );
      stdout = r.stdout; stderr = r.stderr; code = r.code;
    } catch (spawnErr) {
      // spawn-level failures (ENOENT cwd, missing python3 binary, etc.)
      throw new Error(
        `research_graph package not importable from ${settings.packageDir}. ` +
        `Set pluginConfig.packageDir or RESEARCH_GRAPH_PACKAGE_DIR. ` +
        `Underlying error: ${spawnErr.message || String(spawnErr)}`,
      );
    }
    if (code !== 0) {
      throw new Error(
        `research_graph package not importable from ${settings.packageDir}. ` +
        `Set pluginConfig.packageDir or RESEARCH_GRAPH_PACKAGE_DIR. ` +
        `Underlying error: ${stderr.trim() || stdout.trim() || "(no output)"}`,
      );
    }
    settings.__preflightOk = true;
  })();
  try {
    await settings.__preflightPromise;
  } finally {
    if (!settings.__preflightOk) {
      // Allow retries after a failure (e.g. operator fixes packageDir).
      settings.__preflightPromise = null;
    }
  }
}

function pythonMainArgs(settings, cliArgs) {
  return ["-m", "research_graph", "--db", settings.dbPath, ...cliArgs];
}

function helperArgs(settings, helperCliArgs) {
  const helperPath = path.join(__dirname, "plugin_helper.py");
  return [helperPath, "--db", settings.dbPath, ...helperCliArgs];
}

// ---------------------------------------------------------------------------
// Low-level handlers. Each is directly importable by tests.

export const handlers = {
  async memoryStatus({ settings }) {
    // memory_status is the canonical operator probe — explicitly call
    // preflight here so /memory-status surfaces a clear "package not
    // importable" error instead of bubbling a raw ImportError.
    await preflightCheck(settings);
    return runPythonJson(
      settings.python,
      helperArgs(settings, ["status"]),
      { packageDir: settings.packageDir },
    );
  },

  async researchTopic({ settings, topic, question, forceRefresh = false }) {
    if (!topic || !question) {
      throw new Error("research_topic requires both `topic` and `question`");
    }
    if (typeof topic === "string" && topic.includes("::")) {
      throw new Error(
        `research_topic: topic must not contain '::' (got ${JSON.stringify(topic)}). ` +
        "Topics are separated from questions by '::' in /research, so '::' is reserved. " +
        "Rename the topic; questions are allowed to contain '::' verbatim.",
      );
    }
    await preflightCheck(settings);
    const args = helperArgs(settings, [
      "research-topic",
      "--topic", topic,
      "--question", question,
      ...(forceRefresh ? ["--force-refresh"] : []),
    ]);
    return runPythonJson(settings.python, args, {
      packageDir: settings.packageDir,
    });
  },

  // iter-5 continual-research journey entrypoint. Delegates to the
  // plugin_helper `research-journey` command, which wraps the
  // Orchestrator with before/after memory snapshots and returns a
  // rich narrative payload. This is the preferred way for a plugin
  // caller (OpenClaw) to start or resume a research objective —
  // `research_topic` is kept around for thin one-pass calls that
  // only need the raw ResearchResult.
  async researchJourney({
    settings, topic, question,
    skipExternal = false, forceRefresh = false,
  }) {
    if (!topic || !question) {
      throw new Error("research_journey requires both `topic` and `question`");
    }
    if (typeof topic === "string" && topic.includes("::")) {
      throw new Error(
        `research_journey: topic must not contain '::' (got ${JSON.stringify(topic)}). ` +
        "Topics are separated from questions by '::' in /research, so '::' is reserved.",
      );
    }
    await preflightCheck(settings);
    const args = helperArgs(settings, [
      "research-journey",
      "--topic", topic,
      "--question", question,
      ...(skipExternal ? ["--skip-external"] : []),
      ...(forceRefresh ? ["--force-refresh"] : []),
    ]);
    return runPythonJson(settings.python, args, {
      packageDir: settings.packageDir,
    });
  },

  // iter-5 supersession-chain walk. Returns `{chain, siblings, ...}`
  // so OpenClaw can surface "how did this answer evolve" views
  // without having to walk edge labels itself. Accepts EITHER
  // objectiveId OR (topic, question) for resolution; the helper
  // is read-only and returns `{exists: false, ...}` when the
  // (topic, question) pair has no objective yet.
  async thinkingHistory({ settings, objectiveId, topic, question }) {
    if (objectiveId == null && (!topic || !question)) {
      throw new Error(
        "thinking_history requires either `objective_id` or both `topic` and `question`",
      );
    }
    await preflightCheck(settings);
    const cliArgs = ["thinking-history"];
    if (objectiveId != null) {
      cliArgs.push("--objective-id", String(objectiveId));
    }
    if (topic) cliArgs.push("--topic", topic);
    if (question) cliArgs.push("--question", question);
    const args = helperArgs(settings, cliArgs);
    return runPythonJson(settings.python, args, {
      packageDir: settings.packageDir,
    });
  },

  async memorySearch({ settings, query, kind = "sources", topK = 5, objectiveId }) {
    if (!query) throw new Error("memory_search requires `query`");
    if (kind !== "sources" && kind !== "thinkings") {
      throw new Error(`memory_search kind must be sources|thinkings, got ${kind}`);
    }
    await preflightCheck(settings);
    const args = pythonMainArgs(settings, [
      "query",
      query,
      "--kind", kind,
      "--top-k", String(topK),
      "--json",
      ...(objectiveId != null ? ["--objective", String(objectiveId)] : []),
    ]);
    const hits = await runPythonJson(settings.python, args, {
      packageDir: settings.packageDir,
    });
    return { kind, query, topK, hits: hits || [] };
  },

  async memoryIngest({ settings, objectiveId, sources }) {
    if (objectiveId == null) throw new Error("memory_ingest requires `objective_id`");
    if (!Array.isArray(sources)) throw new Error("memory_ingest requires `sources` array");
    await preflightCheck(settings);
    const payload = JSON.stringify({ objective_id: objectiveId, sources });
    const args = helperArgs(settings, ["ingest", "--payload", payload]);
    return runPythonJson(settings.python, args, {
      packageDir: settings.packageDir,
    });
  },

  async graphWalk({ settings, mode, anchorId }) {
    // Note: the spec mentions a "supersession-chain" walk, but the current
    // openclaw-research-graph CLI exposes only 4hop, cocited, and
    // cross-topic-sources. We expose those three as first-class modes and
    // return a structured error for unsupported modes (documented in README).
    const allowed = new Set(["4hop", "cocited", "cross-topic-sources"]);
    if (!allowed.has(mode)) {
      throw new Error(
        `graph_walk mode must be one of ${[...allowed].join(", ")} (got ${mode}). See README for the supersession-chain deviation note.`,
      );
    }
    await preflightCheck(settings);
    const base = ["walk", mode];
    const extras = [];
    if (mode === "4hop" || mode === "cocited") {
      if (anchorId == null) {
        throw new Error(`graph_walk mode ${mode} requires anchorId (a thinking id)`);
      }
      extras.push("--thinking", String(anchorId));
    }
    extras.push("--json");
    const args = pythonMainArgs(settings, [...base, ...extras]);
    const result = await runPythonJson(settings.python, args, {
      packageDir: settings.packageDir,
    });
    return { mode, anchorId: anchorId ?? null, result };
  },
};

// ---------------------------------------------------------------------------
// Slash command parsers exported for direct testing.

/**
 * Parse the raw argv for `/research <topic>::<question>` and dispatch to
 * the iter-5 research journey handler. The first `::` is the separator
 * (use indexOf, not split): topics MUST NOT contain `::`, but questions
 * MAY contain `::` verbatim. iter-5 changed the dispatch target from
 * `researchTopic` (lean one-pass) to `researchJourney` (memory before /
 * run / memory after / delta) so the default slash-command experience
 * reflects continual-memory semantics instead of an opaque
 * ResearchResult blob.
 */
export async function parseAndRunResearchCommand(settings, argv) {
  const raw = (Array.isArray(argv) ? argv.join(" ") : String(argv || "")).trim();
  const sepIdx = raw.indexOf("::");
  if (sepIdx < 0) {
    throw new Error(
      "usage: /research <topic>::<question> (missing '::' separator)",
    );
  }
  const topic = raw.slice(0, sepIdx).trim();
  const question = raw.slice(sepIdx + 2).trim();
  if (!topic) {
    throw new Error("usage: /research <topic>::<question> (empty topic)");
  }
  if (!question) {
    throw new Error("usage: /research <topic>::<question> (empty question)");
  }
  if (topic.includes("::")) {
    // Defensive: by construction this cannot happen because we sliced on the
    // first occurrence, but assert it so a future refactor cannot regress.
    throw new Error(
      `/research topic must not contain '::' (got ${JSON.stringify(topic)}). ` +
      "Rename the topic or escape it; questions may contain '::' but topics may not.",
    );
  }
  return handlers.researchJourney({ settings, topic, question });
}

// ---------------------------------------------------------------------------
// Helpers for formatting tool results into the SDK's expected content shape.

function textContent(payload) {
  return { content: [{ type: "text", text: JSON.stringify(payload, null, 2) }] };
}

function errorContent(err) {
  return {
    isError: true,
    content: [{ type: "text", text: `research-memory error: ${err?.message || String(err)}` }],
  };
}

// ---------------------------------------------------------------------------
// Default export: definePluginEntry. Loading this module outside OpenClaw
// (e.g. in test.js with `import plugin from "./index.js"`) should NOT fail,
// so the SDK import is lazy and wrapped in a try/catch. The handlers export
// is always available for direct invocation.

async function loadDefinePluginEntry() {
  try {
    const mod = await import("openclaw/plugin-sdk/plugin-entry");
    return mod.definePluginEntry;
  } catch {
    return null;
  }
}

async function buildEntry() {
  const definePluginEntry = await loadDefinePluginEntry();
  if (!definePluginEntry) {
    // Minimal stand-in used only when the plugin is imported outside a real
    // OpenClaw host (for example from test.js). Tests don't exercise this
    // path; they call `handlers.*` directly.
    return {
      id: "research-memory",
      name: "Research Memory",
      handlers,
      register() {},
    };
  }

  // typebox is provided by the SDK's sibling dependency set in a real install.
  let Type;
  try {
    const tb = await import("@sinclair/typebox");
    Type = tb.Type;
  } catch {
    Type = null;
  }

  const paramSchema = (obj) => (Type ? Type.Object(obj) : { type: "object", properties: obj });
  const str = () => (Type ? Type.String() : { type: "string" });
  const intT = () => (Type ? Type.Integer() : { type: "integer" });
  const opt = (s) => (Type ? Type.Optional(s) : s);

  return definePluginEntry({
    id: "research-memory",
    name: "Research Memory",
    description:
      "Long-term research memory + orchestration for OpenClaw, backed by the openclaw-research-graph Python graph backend.",
    register(api) {
      const pluginConfig = api.pluginConfig || {};

      const wrapTool = (fn) => async (_id, params) => {
        try {
          const settings = resolveSettings(pluginConfig);
          const payload = await fn(settings, params || {});
          return textContent(payload);
        } catch (err) {
          api.logger?.error?.("research-memory tool failed", { err: String(err) });
          return errorContent(err);
        }
      };

      api.registerTool({
        name: "research_topic",
        description:
          "Create (or reuse) a research objective for a (topic, question) pair and run one research cycle against the canonical research-graph memory. Returns the ResearchResult JSON. For long-term memory flows prefer `research_journey`, which adds before/after memory snapshots and a delta narrative.",
        parameters: paramSchema({
          topic: str(),
          question: str(),
          force_refresh: opt({ type: "boolean" }),
        }),
        execute: wrapTool((settings, p) =>
          handlers.researchTopic({
            settings,
            topic: p.topic,
            question: p.question,
            forceRefresh: !!p.force_refresh,
          }),
        ),
      });

      // iter-5: the continual-research headline tool. OpenClaw hits
      // this for "start or resume research on topic X / question Y"
      // and gets back:
      //   - topic/objective existence flags (was this new or resumed?)
      //   - memory_before / memory_after (live tip + siblings)
      //   - run (raw ResearchResult from the orchestrator)
      //   - delta (quality_overall_delta, outcome, narrative)
      api.registerTool({
        name: "research_journey",
        description:
          "Start or resume a long-term research objective. Resolves or creates the (topic, question) pair, snapshots live memory before and after a research cycle, and returns a rich delta payload { topic, objective, memory_before, run, memory_after, delta } suitable for direct narrative rendering. Use this instead of `research_topic` for any continual-memory flow — it reflects iter-4 quality-gated supersession and hypothesis branching so a weaker run is kept as a sibling branch without unseating the live answer.",
        parameters: paramSchema({
          topic: str(),
          question: str(),
          skip_external: opt({ type: "boolean" }),
          force_refresh: opt({ type: "boolean" }),
        }),
        execute: wrapTool((settings, p) =>
          handlers.researchJourney({
            settings,
            topic: p.topic,
            question: p.question,
            skipExternal: !!p.skip_external,
            forceRefresh: !!p.force_refresh,
          }),
        ),
      });

      api.registerTool({
        name: "thinking_history",
        description:
          "Walk the supersession chain from the current live tip of an objective down to its oldest ancestor, plus enumerate any sibling (branched / hypothesis) live thinkings. Pass either `objective_id` or `(topic, question)`. Read-only — never creates state. Returns { chain, siblings, warnings } where `chain[0]` is the live tip.",
        parameters: paramSchema({
          objective_id: opt(intT()),
          topic: opt(str()),
          question: opt(str()),
        }),
        execute: wrapTool((settings, p) =>
          handlers.thinkingHistory({
            settings,
            objectiveId: p.objective_id,
            topic: p.topic,
            question: p.question,
          }),
        ),
      });

      api.registerTool({
        name: "memory_search",
        description:
          "Search the research-graph memory for ranked retrieval hits. kind=sources (raw documents) or kind=thinkings (distilled conclusions).",
        parameters: paramSchema({
          query: str(),
          kind: opt(str()),
          top_k: opt(intT()),
          objective_id: opt(intT()),
        }),
        execute: wrapTool((settings, p) =>
          handlers.memorySearch({
            settings,
            query: p.query,
            kind: p.kind || "sources",
            topK: p.top_k ?? 5,
            objectiveId: p.objective_id,
          }),
        ),
      });

      api.registerTool({
        name: "memory_ingest",
        description:
          "Ingest raw sources directly into an existing objective, bypassing external search. Used when an agent already has grounded material to store.",
        parameters: paramSchema({
          objective_id: intT(),
          sources: Type ? Type.Array(Type.Object({
            url: opt(str()),
            title: opt(str()),
            content: str(),
          })) : { type: "array" },
        }),
        execute: wrapTool((settings, p) =>
          handlers.memoryIngest({
            settings,
            objectiveId: p.objective_id,
            sources: p.sources,
          }),
        ),
      });

      api.registerTool({
        name: "graph_walk",
        description:
          "Dispatch to a research-graph walk mode: 4hop | cocited | cross-topic-sources. Note: supersession-chain from the spec is not yet exposed by the Python CLI; see plugin README for the deviation note.",
        parameters: paramSchema({
          mode: str(),
          anchor_id: opt(intT()),
        }),
        execute: wrapTool((settings, p) =>
          handlers.graphWalk({ settings, mode: p.mode, anchorId: p.anchor_id }),
        ),
      });

      api.registerTool({
        name: "memory_status",
        description:
          "Return counts of topics, objectives, sources, thinkings, supersessions, and reuses in the canonical research-graph memory.",
        parameters: paramSchema({}),
        execute: wrapTool((settings) => handlers.memoryStatus({ settings })),
      });

      // --- slash commands ---

      const runCommand = async (argv, fn) => {
        const settings = resolveSettings(pluginConfig);
        try {
          const payload = await fn(settings, argv);
          return textContent(payload);
        } catch (err) {
          return errorContent(err);
        }
      };

      api.registerCommand?.({
        name: "research",
        description:
          "Start or resume a long-term research objective. Usage: /research <topic>::<question>. The topic must NOT contain '::' (the first '::' is the separator); the question MAY contain '::' verbatim. Returns a continual-memory journey payload (memory_before, run, memory_after, delta) — iter-5 default.",
        async execute({ argv }) {
          return runCommand(argv, async (settings, a) => {
            return parseAndRunResearchCommand(settings, a);
          });
        },
      });

      api.registerCommand?.({
        name: "research-history",
        description:
          "Show the supersession chain + sibling hypothesis branches for a research objective. Usage: /research-history <topic>::<question>",
        async execute({ argv }) {
          return runCommand(argv, async (settings, a) => {
            const raw = (Array.isArray(a) ? a.join(" ") : String(a || "")).trim();
            const sepIdx = raw.indexOf("::");
            if (sepIdx < 0) {
              throw new Error("usage: /research-history <topic>::<question>");
            }
            const topic = raw.slice(0, sepIdx).trim();
            const question = raw.slice(sepIdx + 2).trim();
            if (!topic || !question) {
              throw new Error("usage: /research-history <topic>::<question>");
            }
            return handlers.thinkingHistory({ settings, topic, question });
          });
        },
      });

      api.registerCommand?.({
        name: "memory-search",
        description:
          "Search research-graph memory. Usage: /memory-search <sources|thinkings>::<query>",
        async execute({ argv }) {
          return runCommand(argv, async (settings, a) => {
            const raw = (Array.isArray(a) ? a.join(" ") : String(a || "")).trim();
            const [kind, ...rest] = raw.split("::");
            const query = rest.join("::").trim();
            if (!kind || !query) {
              throw new Error("usage: /memory-search <sources|thinkings>::<query>");
            }
            return handlers.memorySearch({ settings, query, kind: kind.trim() });
          });
        },
      });

      api.registerCommand?.({
        name: "memory-status",
        description: "Show research-graph memory counts.",
        async execute() {
          return runCommand([], (settings) => handlers.memoryStatus({ settings }));
        },
      });

      api.registerCommand?.({
        name: "graph-walk",
        description:
          "Run a research-graph walk. Usage: /graph-walk <4hop|cocited|cross-topic-sources> [anchor_id]",
        async execute({ argv }) {
          return runCommand(argv, async (settings, a) => {
            const raw = (Array.isArray(a) ? a.join(" ") : String(a || "")).trim();
            const [mode, anchorStr] = raw.split(/\s+/);
            if (!mode) throw new Error("usage: /graph-walk <mode> [anchor_id]");
            const anchorId = anchorStr ? Number.parseInt(anchorStr, 10) : undefined;
            return handlers.graphWalk({ settings, mode, anchorId });
          });
        },
      });
    },
  });
}

const entryPromise = buildEntry();
const entry = await entryPromise;
export default entry;
