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
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(
          `python bridge exited ${code}: ${stderr.trim() || stdout.trim()}`,
        ));
      }
    });
  });
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
    const { stdout } = await runPython(
      settings.python,
      helperArgs(settings, ["status"]),
      { packageDir: settings.packageDir },
    );
    return parseJson(stdout);
  },

  async researchTopic({ settings, topic, question, forceRefresh = false }) {
    if (!topic || !question) {
      throw new Error("research_topic requires both `topic` and `question`");
    }
    const args = helperArgs(settings, [
      "research-topic",
      "--topic", topic,
      "--question", question,
      ...(forceRefresh ? ["--force-refresh"] : []),
    ]);
    const { stdout } = await runPython(settings.python, args, {
      packageDir: settings.packageDir,
    });
    return parseJson(stdout);
  },

  async memorySearch({ settings, query, kind = "sources", topK = 5, objectiveId }) {
    if (!query) throw new Error("memory_search requires `query`");
    if (kind !== "sources" && kind !== "thinkings") {
      throw new Error(`memory_search kind must be sources|thinkings, got ${kind}`);
    }
    const args = pythonMainArgs(settings, [
      "query",
      query,
      "--kind", kind,
      "--top-k", String(topK),
      "--json",
      ...(objectiveId != null ? ["--objective", String(objectiveId)] : []),
    ]);
    const { stdout } = await runPython(settings.python, args, {
      packageDir: settings.packageDir,
    });
    return { kind, query, topK, hits: parseJson(stdout) || [] };
  },

  async memoryIngest({ settings, objectiveId, sources }) {
    if (objectiveId == null) throw new Error("memory_ingest requires `objective_id`");
    if (!Array.isArray(sources)) throw new Error("memory_ingest requires `sources` array");
    const payload = JSON.stringify({ objective_id: objectiveId, sources });
    const args = helperArgs(settings, ["ingest", "--payload", payload]);
    const { stdout } = await runPython(settings.python, args, {
      packageDir: settings.packageDir,
    });
    return parseJson(stdout);
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
    const { stdout } = await runPython(settings.python, args, {
      packageDir: settings.packageDir,
    });
    return { mode, anchorId: anchorId ?? null, result: parseJson(stdout) };
  },
};

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
          "Create (or reuse) a research objective for a (topic, question) pair and run one research cycle against the canonical research-graph memory. Returns the ResearchResult JSON.",
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
          "Start or resume a research objective. Usage: /research <topic>::<question>",
        async execute({ argv }) {
          return runCommand(argv, async (settings, a) => {
            const raw = (Array.isArray(a) ? a.join(" ") : String(a || "")).trim();
            const [topic, ...rest] = raw.split("::");
            const question = rest.join("::").trim();
            if (!topic || !question) {
              throw new Error("usage: /research <topic>::<question>");
            }
            return handlers.researchTopic({ settings, topic: topic.trim(), question });
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
