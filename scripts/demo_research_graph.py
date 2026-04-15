#!/usr/bin/env python3
"""Runnable narrative demo of the research_graph package against the
InMemoryGraphBackend.

Builds a non-trivial multi-topic graph, drives it through a simulated
5-day evolution, and prints the supersession chain + four-hop walk
result at the end. No arguments, no network, stdlib-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_graph import store  # noqa: E402
from research_graph.graph import InMemoryGraphBackend  # noqa: E402
from research_graph.orchestrator import Orchestrator  # noqa: E402
from research_graph.search import SearchHit  # noqa: E402


class DemoSearch:
    def __init__(self, script: dict[int, list[SearchHit]]) -> None:
        self._script = script
        self.current_day = 1

    def search(self, query: str) -> list[SearchHit]:
        return list(self._script.get(self.current_day, []))


def _hr(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> int:
    backend = InMemoryGraphBackend()

    # Three topics, six objectives.
    t_speed = store.create_topic(backend, "local-llm-mac-mini")
    t_mem = store.create_topic(backend, "agentic-memory")
    t_search = store.create_topic(backend, "vector-search")

    o_speed_tok = store.create_objective(backend, t_speed, "fastest local llm on 16gb mac mini")
    o_speed_q = store.create_objective(backend, t_speed, "best instruction following local llm")
    o_mem_a = store.create_objective(backend, t_mem, "best agentic memory architecture")
    o_mem_b = store.create_objective(backend, t_mem, "long term memory for llm agents")
    o_search_a = store.create_objective(backend, t_search, "best embeddings for local search")
    o_search_b = store.create_objective(backend, t_search, "vector index tradeoffs mac mini")

    speed_script = {
        1: [
            SearchHit("https://ex/llama", "Llama3", "Llama 3.1 8B Q4 about 18 tokens per second on 16gb mac mini metal"),
            SearchHit("https://ex/mistral", "Mistral", "Mistral 7B Q4 about 22 tokens per second mac mini apple silicon"),
        ],
        2: [
            SearchHit("https://ex/llama", "Llama3", "Llama 3.1 8B Q4 about 18 tokens per second on 16gb mac mini metal"),
            SearchHit("https://ex/phi", "Phi3", "Phi 3 mini Q4 about 35 tokens per second on 16gb mac mini"),
        ],
        3: [
            SearchHit("https://ex/qwen", "Qwen 2026", "Qwen 2.5 7B instruct outperforms Llama 3.1 on mac mini"),
        ],
        4: [
            SearchHit("https://ex/qwen", "Qwen 2026", "Qwen 2.5 7B instruct outperforms Llama 3.1 on mac mini"),
        ],
        5: [
            SearchHit("https://ex/gemma", "Gemma2", "Gemma 2 9B quantized about 15 tokens per second mac mini metal"),
        ],
    }
    quality_script = {
        1: [SearchHit("https://ex/llama-q", "llama instruct", "Llama 3.1 instruct follows multi step instructions")],
        2: [SearchHit("https://ex/qwen-q", "qwen instruct", "Qwen 2.5 instruct answers multi step questions accurately")],
        3: [SearchHit("https://ex/qwen-q", "qwen instruct", "Qwen 2.5 instruct answers multi step questions accurately")],
        4: [SearchHit("https://ex/phi-q", "phi instruct", "Phi 3 mini instruct handles short instructions")],
        5: [SearchHit("https://ex/phi-q", "phi instruct", "Phi 3 mini instruct handles short instructions")],
    }
    mem_a_script = {
        1: [SearchHit("https://ex/memgpt", "memgpt", "MemGPT hierarchical memory for LLM agents archival store")],
        2: [SearchHit("https://ex/letta", "letta", "Letta productized successor to MemGPT exposing REST API")],
        3: [SearchHit("https://ex/letta", "letta", "Letta productized successor to MemGPT exposing REST API")],
        4: [SearchHit("https://ex/zep", "zep", "Zep long term memory temporal knowledge graph for agents")],
        5: [SearchHit("https://ex/zep", "zep", "Zep long term memory temporal knowledge graph for agents")],
    }
    mem_b_script = {
        1: [SearchHit("https://ex/memgpt", "memgpt", "MemGPT hierarchical memory for LLM agents archival store")],
        2: [SearchHit("https://ex/memgpt", "memgpt", "MemGPT hierarchical memory for LLM agents archival store")],
        3: [SearchHit("https://ex/graphmem", "graphmem", "Knowledge graph backed memory for llm agents temporal")],
        4: [SearchHit("https://ex/graphmem", "graphmem", "Knowledge graph backed memory for llm agents temporal")],
        5: [SearchHit("https://ex/summary", "summary", "Rolling summary memory compaction for long chat histories")],
    }
    search_a_script = {
        1: [SearchHit("https://ex/bge", "bge", "BGE small embeddings offer strong recall on english text")],
        2: [SearchHit("https://ex/mxbai", "mxbai", "mxbai large v1 embeddings dominate the MTEB leaderboard")],
        3: [SearchHit("https://ex/mxbai", "mxbai", "mxbai large v1 embeddings dominate the MTEB leaderboard")],
        4: [SearchHit("https://ex/nomic", "nomic", "Nomic embed text v1.5 matryoshka embedding dimensions")],
        5: [SearchHit("https://ex/nomic", "nomic", "Nomic embed text v1.5 matryoshka embedding dimensions")],
    }
    search_b_script = {
        1: [SearchHit("https://ex/faiss", "faiss", "FAISS offers IVF HNSW and flat indexes on mac mini")],
        2: [SearchHit("https://ex/faiss", "faiss", "FAISS offers IVF HNSW and flat indexes on mac mini")],
        3: [SearchHit("https://ex/hnswlib", "hnswlib", "hnswlib small cpu index fast approximate nearest neighbor")],
        4: [SearchHit("https://ex/hnswlib", "hnswlib", "hnswlib small cpu index fast approximate nearest neighbor")],
        5: [SearchHit("https://ex/annoy", "annoy", "Annoy spotify approximate nearest neighbors mmap index")],
    }

    scripts = {
        o_speed_tok: DemoSearch(speed_script),
        o_speed_q: DemoSearch(quality_script),
        o_mem_a: DemoSearch(mem_a_script),
        o_mem_b: DemoSearch(mem_b_script),
        o_search_a: DemoSearch(search_a_script),
        o_search_b: DemoSearch(search_b_script),
    }

    for day in range(1, 6):
        _hr(f"Day {day}")
        for oid, backend_search in scripts.items():
            backend_search.current_day = day
            force = day > 1
            orch = Orchestrator(backend, search=backend_search)
            result = orch.research(oid, force_refresh=force)
            tag = result.new_thinking_id or f"reuse={result.reused_thinking_id}"
            print(
                f"  obj {oid}: mode={result.mode} "
                f"new_srcs={len(result.new_source_ids)} "
                f"thinking={tag}"
            )

    _hr("Final thinking chain for objective 'fastest local llm on 16gb mac mini'")
    chain = store.list_thinkings(backend, o_speed_tok)
    for row in chain:
        parent = row.get("supersedes_id")
        arrow = f" -> supersedes {parent}" if parent else ""
        snippet = row["content"].splitlines()[0][:60]
        print(f"  thinking {row['id']}{arrow}: {snippet}")

    _hr("Supersession chain walk from latest speed thinking")
    if chain:
        latest = int(chain[-1]["id"])
        walk_chain = store.supersession_chain(backend, latest)
        print("  " + " -> ".join(str(int(c["id"])) for c in walk_chain))

    _hr("4-hop evidence walk from latest speed thinking")
    if chain:
        latest = int(chain[-1]["id"])
        walk = store.four_hop_evidence_walk(backend, latest)
        print(f"  start              = {int(walk['start']['id'])}")
        print(f"  shared_sources     = {[int(s['id']) for s in walk['shared_sources']]}")
        print(f"  related_thinkings  = {[int(t['id']) for t in walk['related_thinkings']]}")
        print(f"  new_sources        = {[int(s['id']) for s in walk['new_sources']]}")

    _hr("Cross-topic shared sources")
    xs = store.cross_topic_shared_sources(backend)
    if xs:
        for row in xs:
            print(f"  hash={row['content_hash'][:12]}... topics={row['topic_ids']}")
    else:
        print("  (none)")

    n_sources = sum(
        len(store.list_sources(backend, int(o["id"])))
        for o in store.list_objectives(backend)
    )
    n_thinkings = sum(
        len(store.list_thinkings(backend, int(o["id"])))
        for o in store.list_objectives(backend)
    )
    all_reuses = store.list_reuses(backend)
    supersession_rows = [
        t for t in (
            th
            for o in store.list_objectives(backend)
            for th in store.list_thinkings(backend, int(o["id"]))
        )
        if t.get("supersedes_id")
    ]
    print()
    print(
        f"demo OK - {n_sources} sources, {n_thinkings} thinkings, "
        f"{len(supersession_rows)} supersessions, {len(all_reuses)} dedup-reuses"
    )
    print(
        f"graph: {backend.node_count} nodes, {backend.edge_count} edges, "
        f"{len(all_reuses)} reuses, {len(supersession_rows)} supersessions"
    )

    if n_sources < 4:
        print(f"FAIL: expected >=4 sources, got {n_sources}", file=sys.stderr)
        return 1
    if n_thinkings < 3:
        print(f"FAIL: expected >=3 thinkings, got {n_thinkings}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
