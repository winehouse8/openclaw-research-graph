#!/usr/bin/env python3
"""Runnable narrative demo of the research_graph package.

Walks a simulated 5-day evolution of a single research objective using a
fake timed search backend, and prints the supersession chain forming
live. No arguments, no network, stdlib-only; uses a tempdir DB and
tears itself down at exit.

    python3 scripts/demo_research_graph.py

The script exits non-zero if the final counts disagree with expectation.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running from repo root without install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_graph import storage  # noqa: E402
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
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "demo.db"
        conn = storage.connect(db_path)

        topic_id = storage.create_topic(conn, "local-llm-mac-mini")
        obj_id = storage.create_objective(
            conn, topic_id, "fastest local llm on 16gb mac mini"
        )

        script = {
            1: [
                SearchHit("https://ex/llama", "Llama3 bench",
                          "Llama 3.1 8B Q4 runs at about 18 tokens per second on 16gb mac mini metal"),
                SearchHit("https://ex/mistral", "Mistral bench",
                          "Mistral 7B Q4 reaches roughly 22 tokens per second on mac mini apple silicon"),
            ],
            2: [
                SearchHit("https://ex/llama", "Llama3 bench",
                          "Llama 3.1 8B Q4 runs at about 18 tokens per second on 16gb mac mini metal"),
                SearchHit("https://ex/phi", "Phi3 bench",
                          "Phi 3 mini Q4 runs at around 35 tokens per second on 16gb mac mini hardware"),
            ],
            3: [
                SearchHit("https://ex/qwen", "Qwen 2026",
                          "Qwen 2.5 7B instruct Q4 K M outperforms Llama 3.1 on 2026 mac mini benchmarks"),
            ],
            4: [
                # exact repeat of day 3
                SearchHit("https://ex/qwen", "Qwen 2026",
                          "Qwen 2.5 7B instruct Q4 K M outperforms Llama 3.1 on 2026 mac mini benchmarks"),
            ],
            5: [
                SearchHit("https://ex/gemma", "Gemma2 bench",
                          "Gemma 2 9B quantized runs at about 15 tokens per second on mac mini metal backend"),
            ],
        }

        backend = DemoSearch(script)
        orch = Orchestrator(conn, search=backend)

        supersession_count = 0
        dedup_reuses = 0
        for day in range(1, 6):
            _hr(f"Day {day}")
            backend.current_day = day
            force = day > 1
            result = orch.research(obj_id, force_refresh=force)
            print(f"  mode             = {result.mode}")
            print(f"  new_source_ids   = {result.new_source_ids}")
            print(f"  reused_source_ids= {result.reused_source_ids}")
            if result.new_thinking_id is not None:
                print(f"  new_thinking_id  = {result.new_thinking_id}")
                if result.supersedes_id is not None:
                    supersession_count += 1
                    print(f"  supersedes       = {result.supersedes_id}")
            else:
                print(f"  reused_thinking  = {result.reused_thinking_id}")
                dedup_reuses += 1

        _hr("Final thinking chain")
        chain = storage.list_thinkings(conn, obj_id)
        for row in chain:
            parent = row["supersedes_id"]
            arrow = f" -> supersedes {parent}" if parent else ""
            snippet = row["content"].splitlines()[0][:60]
            print(f"  thinking {row['id']}{arrow}: {snippet}")

        _hr("Final source corpus")
        for row in storage.list_sources(conn, obj_id):
            print(f"  source {row['id']}: {row['url']}")

        n_sources = len(storage.list_sources(conn, obj_id))
        n_thinkings = len(chain)
        conn.close()

        print()
        print(
            f"demo OK - {n_sources} sources, {n_thinkings} thinkings, "
            f"{supersession_count} supersessions, {dedup_reuses} dedup-reuses"
        )

        # sanity guardrails so the demo doubles as a smoke test
        if n_sources < 4:
            print(f"FAIL: expected >=4 sources, got {n_sources}", file=sys.stderr)
            return 1
        if n_thinkings < 3:
            print(f"FAIL: expected >=3 thinkings, got {n_thinkings}", file=sys.stderr)
            return 1
        if supersession_count < 2:
            print(f"FAIL: expected >=2 supersessions, got {supersession_count}", file=sys.stderr)
            return 1
        if dedup_reuses < 1:
            print(f"FAIL: expected >=1 dedup reuse, got {dedup_reuses}", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
