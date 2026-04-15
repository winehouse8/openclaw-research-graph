// End-to-end tests for the research-memory plugin.
//
// These tests import the plugin's handlers directly and drive them against a
// tempdir-backed research_graph JSON db, without requiring a live OpenClaw
// daemon. They verify the five tool surfaces actually bridge to the real
// Python graph backend and return structured data.
//
// Run: node --test test.js

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { handlers, resolveSettings } from "./index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..", "..");

function makeTempSettings() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "research-memory-test-"));
  const dbPath = path.join(tmp, "research_graph.json");
  const settings = resolveSettings(
    {},
    { dbPath, python: process.env.PYTHON || "python3", packageDir: REPO_ROOT },
  );
  return { tmp, settings };
}

test("memory_status on an empty db returns zero counts", async () => {
  const { tmp, settings } = makeTempSettings();
  try {
    const status = await handlers.memoryStatus({ settings });
    assert.ok(status && status.counts, "status must include counts");
    assert.equal(status.counts.topics, 0);
    assert.equal(status.counts.objectives, 0);
    assert.equal(status.counts.sources, 0);
    assert.equal(status.counts.thinkings, 0);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("research_topic creates topic+objective and runs one research cycle", async () => {
  const { tmp, settings } = makeTempSettings();
  try {
    const before = await handlers.memoryStatus({ settings });
    const result = await handlers.researchTopic({
      settings,
      topic: "quantum-computing",
      question: "What is the current status of post-quantum cryptography?",
    });

    // Shape checks
    assert.equal(result.topic, "quantum-computing");
    assert.equal(
      result.question,
      "What is the current status of post-quantum cryptography?",
    );
    assert.ok(Number.isInteger(result.topic_id), "topic_id must be an int");
    assert.ok(Number.isInteger(result.objective_id), "objective_id must be an int");
    assert.ok(result.research, "research sub-payload must exist");
    assert.ok(
      result.research.mode === "cold_start" || result.research.mode === "memory_augmented",
      `unexpected research mode: ${result.research.mode}`,
    );
    assert.ok(Array.isArray(result.research.new_source_ids));

    const after = await handlers.memoryStatus({ settings });
    assert.ok(
      after.counts.topics >= before.counts.topics + 1,
      "topics should increase after research_topic",
    );
    assert.ok(
      after.counts.objectives >= before.counts.objectives + 1,
      "objectives should increase after research_topic",
    );
    assert.ok(
      after.counts.sources >= before.counts.sources,
      "sources should not decrease",
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("memory_search finds ingested content and returns a ranked hit list", async () => {
  const { tmp, settings } = makeTempSettings();
  try {
    // Seed the graph with a topic + objective + source via research_topic,
    // then add a deterministic extra source via memory_ingest so the search
    // has a known-needle to find.
    const rt = await handlers.researchTopic({
      settings,
      topic: "photonic-computing",
      question: "How practical are photonic tensor cores?",
    });
    const ingest = await handlers.memoryIngest({
      settings,
      objectiveId: rt.objective_id,
      sources: [
        {
          url: "https://example.com/photonic-mach-zehnder",
          title: "Photonic Mach-Zehnder tensor core survey",
          content:
            "photonic mach zehnder tensor core latency photonic photonic photonic integrated circuit",
        },
      ],
    });
    assert.ok(Array.isArray(ingest.inserted) && ingest.inserted.length === 1,
      "memory_ingest should insert exactly one new source");

    const search = await handlers.memorySearch({
      settings,
      query: "photonic mach zehnder",
      kind: "sources",
      topK: 5,
    });
    assert.equal(search.kind, "sources");
    assert.ok(Array.isArray(search.hits), "hits must be an array");
    assert.ok(
      search.hits.length >= 1,
      `expected at least one hit for the ingested photonic source, got ${search.hits.length}`,
    );
    const hit = search.hits[0];
    assert.ok(typeof hit === "object" && hit != null, "hit must be an object");
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("graph_walk 4hop returns a structured result for a real thinking anchor", async () => {
  const { tmp, settings } = makeTempSettings();
  try {
    const rt = await handlers.researchTopic({
      settings,
      topic: "neuromorphic-hardware",
      question: "Which neuromorphic chips are production-ready in 2026?",
    });
    assert.ok(
      Number.isInteger(rt.research?.new_thinking_id),
      "research_topic should emit a new_thinking_id for a cold start",
    );

    const walk = await handlers.graphWalk({
      settings,
      mode: "4hop",
      anchorId: rt.research.new_thinking_id,
    });
    assert.equal(walk.mode, "4hop");
    assert.equal(walk.anchorId, rt.research.new_thinking_id);
    assert.ok(walk.result && typeof walk.result === "object", "walk result must be an object");
    // The 4hop walk result shape is:
    //   { start, shared_sources, related_thinkings, new_sources }
    for (const key of ["start", "shared_sources", "related_thinkings", "new_sources"]) {
      assert.ok(key in walk.result, `4hop walk result missing key: ${key}`);
    }
    assert.ok(Array.isArray(walk.result.shared_sources));
    assert.ok(Array.isArray(walk.result.related_thinkings));
    assert.ok(Array.isArray(walk.result.new_sources));
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("graph_walk rejects unsupported modes with a clear error", async () => {
  const { tmp, settings } = makeTempSettings();
  try {
    await assert.rejects(
      () => handlers.graphWalk({ settings, mode: "supersession-chain", anchorId: 1 }),
      /graph_walk mode must be one of/,
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
