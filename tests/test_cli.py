from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "research_graph", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd or REPO_ROOT),
        check=False,
    )


class CLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "cli.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return run_cli("--db", self.db, *args)

    def test_init_topic_objective_research(self) -> None:
        r = self._run("init")
        self.assertEqual(r.returncode, 0, r.stderr)

        r = self._run("topic", "add", "llm")
        self.assertEqual(r.returncode, 0, r.stderr)
        topic_id = int(r.stdout.strip())
        self.assertGreater(topic_id, 0)

        r = self._run("topic", "list", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        topics = json.loads(r.stdout.strip())
        self.assertEqual(topics[0]["name"], "llm")

        r = self._run("objective", "add", "--topic", "llm", "best local llm mac mini")
        self.assertEqual(r.returncode, 0, r.stderr)
        oid = int(r.stdout.strip())

        r = self._run("research", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout.strip())
        self.assertEqual(result["mode"], "cold_start")
        self.assertGreater(len(result["new_source_ids"]), 0)
        self.assertIsNotNone(result["new_thinking_id"])

        # query sources
        r = self._run("query", "llama mac mini", "--objective", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = json.loads(r.stdout.strip())
        self.assertGreater(len(rows), 0)

        # query thinkings
        r = self._run(
            "query", "synthesis", "--kind", "thinkings", "--objective", str(oid), "--json"
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = json.loads(r.stdout.strip())
        self.assertGreaterEqual(len(rows), 1)

        # source list / delete
        r = self._run("source", "list", "--objective", str(oid), "--json")
        sources = json.loads(r.stdout.strip())
        first_sid = sources[0]["id"]
        r = self._run("source", "delete", str(first_sid))
        self.assertEqual(r.returncode, 0, r.stderr)

        # thinking list
        r = self._run("thinking", "list", "--objective", str(oid), "--json")
        thinkings = json.loads(r.stdout.strip())
        self.assertGreaterEqual(len(thinkings), 1)

    def test_schedule_cron_and_once_idempotent(self) -> None:
        self._run("init")
        self._run("topic", "add", "llm")
        self._run("objective", "add", "--topic", "llm", "best local llm mac mini")

        r = self._run("schedule", "cron", "--hour", "9", "--topic", "llm")
        self.assertEqual(r.returncode, 0, r.stderr)
        line = r.stdout.strip()
        self.assertTrue(line.startswith("0 9 "))
        self.assertIn("schedule once", line)
        self.assertIn("--topic", line)

        r = self._run("schedule", "once", "--topic", "llm", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        first = json.loads(r.stdout.strip())
        self.assertEqual(len(first), 1)
        first_new_sources = first[0]["new_sources"]

        r = self._run("schedule", "once", "--topic", "llm", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        second = json.loads(r.stdout.strip())
        # idempotent: second run produces no new sources due to dedup
        self.assertEqual(second[0]["new_sources"], [])

    def test_api_and_cli_return_identical_dict(self) -> None:
        from research_graph import api

        self._run("init")
        self._run("topic", "add", "llm")
        r = self._run("objective", "add", "--topic", "llm", "best local llm mac mini")
        oid = int(r.stdout.strip())

        # Run CLI subprocess first; its --json output is the canonical dict.
        r = self._run("research", str(oid), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        cli_dict = json.loads(r.stdout.strip())

        # Now call api.research() against the SAME db so dedup collapses
        # the second run; we'll directly compare the canonical contract by
        # invoking research on a fresh objective so both surfaces hit the
        # same code path on identical state.
        r = self._run("objective", "add", "--topic", "llm", "another best llm question")
        oid2 = int(r.stdout.strip())

        api_dict = api.research(self.db, oid2)
        r2 = self._run("research", str(oid2), "--json")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        # api.research() above already mutated the db (cold_start). The CLI
        # call therefore sees memory_augmented mode; what we really test is
        # that whatever dict shape the CLI prints matches a fresh api call
        # against the SAME post-state. So make one more api call and diff.
        api_dict2 = api.research(self.db, oid2)
        cli_dict2 = json.loads(r2.stdout.strip())
        # Compare cli_dict2 to api_dict2: both ran after the first api call,
        # so the state is identical (memory_augmented, no new sources).
        self.assertEqual(set(cli_dict2.keys()), set(api_dict2.keys()))
        self.assertEqual(cli_dict2, api_dict2)
        # And confirm the canonical key set matches the spec exactly.
        # Spec L40, L87 ("축적된 지식을 다시 사용") requires the caller
        # be able to observe what was reused, not just what's new — so
        # `reused_source_ids` and `rejected_reasons` are part of the
        # canonical contract now. Without them, OpenClaw cannot tell
        # "no new evidence arrived" from "the actor was rejected by
        # the critic" or "the run blended N reused sources."
        expected_keys = {
            "objective_id",
            "mode",
            "new_source_ids",
            "reused_source_ids",
            "new_thinking_id",
            "reused_thinking_id",
            "supersedes_id",
            "rejected_reasons",
            # iter-4 continual-research fields (see
            # test_spec_compliance::ResearchResultContractTest).
            "quality_score",
            "actor_backend",
            "branched",
        }
        self.assertEqual(set(cli_dict.keys()), expected_keys)
        self.assertEqual(set(api_dict.keys()), expected_keys)

    def test_objective_list_filtered(self) -> None:
        self._run("init")
        self._run("topic", "add", "a")
        self._run("topic", "add", "b")
        self._run("objective", "add", "--topic", "a", "qa")
        self._run("objective", "add", "--topic", "b", "qb")
        r = self._run("objective", "list", "--topic", "a", "--json")
        rows = json.loads(r.stdout.strip())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question"], "qa")


if __name__ == "__main__":
    unittest.main()
