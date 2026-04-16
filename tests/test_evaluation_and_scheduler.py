"""iter-7: evaluation run-log + scheduler journey-path tests.

Locks in the structural fix where `schedule.run_once_for_topic` routes
through `api.research_journey` + `evaluation.record_run` instead of
calling the raw orchestrator. This is what makes the spec L91 ("매일
오전 9시 자동 리서치") verifiable — every scheduled run now produces a
quality-tracked log entry.

Scenarios:
  1. Cold-start journey → record_run → log file exists + entry valid
  2. quality_trend after multiple runs → correct aggregation
  3. stale_objectives detection
  4. run_once_for_topic → uses journey path + records to log
  5. run_daily_summary → aggregate outcomes
  6. E2E: cold → second run → trend shows stable quality
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from research_graph import api, evaluation, schedule, store
from research_graph.graph import get_default_backend


class RecordRunTest(unittest.TestCase):

    def setUp(self):
        self._old_search = os.environ.get("OPENCLAW_RESEARCH_SEARCH")
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "offline"
        os.environ["OPENCLAW_RESEARCH_ACTOR"] = "placeholder"
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")
        self.log = str(Path(self.tmp.name) / "research_runs.jsonl")

    def tearDown(self):
        self.tmp.cleanup()
        if self._old_search is None:
            os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)
        else:
            os.environ["OPENCLAW_RESEARCH_SEARCH"] = self._old_search

    def test_record_creates_log_and_entry_is_valid(self):
        journey = api.research_journey(self.db, "t1", "q1")
        entry = evaluation.record_run(self.log, journey)
        # Log file exists.
        self.assertTrue(Path(self.log).exists())
        # Entry has the key fields.
        self.assertIsNotNone(entry["timestamp"])
        self.assertEqual(entry["objective_id"], journey["objective"]["id"])
        self.assertEqual(entry["outcome"], "cold_start")
        self.assertIsNotNone(entry["quality_overall"])
        self.assertIn("_raw", entry)

    def test_multiple_records_append(self):
        j1 = api.research_journey(self.db, "t1", "q1")
        j2 = api.research_journey(self.db, "t1", "q1")
        evaluation.record_run(self.log, j1)
        evaluation.record_run(self.log, j2)
        entries = evaluation._read_log(self.log)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["outcome"], "cold_start")
        self.assertIn(entries[1]["outcome"], {"reused", "unchanged"})


class QualityTrendTest(unittest.TestCase):

    def setUp(self):
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "offline"
        os.environ["OPENCLAW_RESEARCH_ACTOR"] = "placeholder"
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")
        self.log = str(Path(self.tmp.name) / "research_runs.jsonl")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)

    def test_trend_after_two_runs(self):
        j1 = api.research_journey(self.db, "t1", "q1")
        j2 = api.research_journey(self.db, "t1", "q1")
        evaluation.record_run(self.log, j1)
        evaluation.record_run(self.log, j2)
        trend = evaluation.quality_trend(
            self.log, j1["objective"]["id"]
        )
        self.assertEqual(trend["run_count"], 2)
        self.assertIsNotNone(trend["first_quality"])
        self.assertIsNotNone(trend["latest_quality"])
        self.assertTrue(trend["improved_or_stable"])
        self.assertEqual(trend["regression_count"], 0)

    def test_trend_empty_log_returns_zero_runs(self):
        trend = evaluation.quality_trend(self.log, 99999)
        self.assertEqual(trend["run_count"], 0)
        self.assertIsNone(trend["first_quality"])


class StaleObjectivesTest(unittest.TestCase):

    def setUp(self):
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "offline"
        os.environ["OPENCLAW_RESEARCH_ACTOR"] = "placeholder"
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")
        self.log = str(Path(self.tmp.name) / "research_runs.jsonl")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)

    def test_recent_run_is_not_stale(self):
        j1 = api.research_journey(self.db, "t1", "q1")
        evaluation.record_run(self.log, j1)
        stale = evaluation.stale_objectives(self.log, max_age_hours=24.0)
        # Just recorded → should NOT be stale.
        self.assertEqual(len(stale), 0)

    def test_ancient_entry_detected_as_stale(self):
        # Manually write an entry with an old timestamp.
        entry = {
            "timestamp": "2020-01-01T00:00:00+00:00",
            "objective_id": 42,
            "question": "old q",
            "topic_name": "t1",
        }
        with open(self.log, "w") as f:
            f.write(json.dumps(entry) + "\n")
        stale = evaluation.stale_objectives(self.log, max_age_hours=1.0)
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["objective_id"], 42)


class SchedulerJourneyPathTest(unittest.TestCase):
    """The critical test: schedule.run_once_for_topic now routes
    through api.research_journey and records to the evaluation log."""

    def setUp(self):
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "offline"
        os.environ["OPENCLAW_RESEARCH_ACTOR"] = "placeholder"
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)

    def test_run_once_uses_journey_and_records_log(self):
        # Seed a topic + objective via journey so there's something
        # for the scheduler to iterate over.
        api.research_journey(self.db, "llm", "best local llm")

        # Now run the scheduler.
        results = schedule.run_once_for_topic(self.db, "llm")
        self.assertEqual(len(results), 1)
        # The result IS a journey dict (not the old minimal dict).
        r = results[0]
        self.assertIn("delta", r)
        self.assertIn("memory_before", r)
        self.assertIn("memory_after", r)
        self.assertIn("outcome", r["delta"])

        # The run was recorded to the evaluation log.
        log_path = evaluation._default_log_path(self.db)
        self.assertTrue(log_path.exists())
        entries = evaluation._read_log(log_path)
        # At least 1 entry from the scheduler run (cold start may
        # have also been recorded if journey does it, but the
        # scheduler definitely adds one).
        scheduler_entries = [
            e for e in entries
            if e.get("question") == "best local llm"
        ]
        self.assertGreaterEqual(len(scheduler_entries), 1)

    def test_run_once_nonexistent_topic_returns_empty(self):
        results = schedule.run_once_for_topic(self.db, "nope")
        self.assertEqual(results, [])

    def test_daily_summary_aggregates(self):
        api.research_journey(self.db, "llm", "q1")
        api.research_journey(self.db, "llm", "q2")
        summary = schedule.run_daily_summary(self.db, "llm")
        self.assertEqual(summary["topic"], "llm")
        self.assertEqual(summary["objectives_run"], 2)
        self.assertIn("outcomes", summary)
        self.assertIn("all_stable_or_improved", summary)
        self.assertTrue(summary["all_stable_or_improved"])


class E2EQualityTrackingTest(unittest.TestCase):
    """Full cycle: journey → scheduler run → quality trend check."""

    def setUp(self):
        os.environ["OPENCLAW_RESEARCH_SEARCH"] = "offline"
        os.environ["OPENCLAW_RESEARCH_ACTOR"] = "placeholder"
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "rg.json")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("OPENCLAW_RESEARCH_SEARCH", None)

    def test_cold_then_cron_then_trend(self):
        # Cold start via journey (not scheduler).
        j1 = api.research_journey(self.db, "llm", "q1")
        log = evaluation._default_log_path(self.db)
        evaluation.record_run(log, j1)

        # Simulate a cron run (uses scheduler → journey → record).
        schedule.run_once_for_topic(self.db, "llm")

        # Check trend.
        trend = evaluation.quality_trend(
            log, j1["objective"]["id"]
        )
        self.assertGreaterEqual(trend["run_count"], 2)
        self.assertTrue(trend["improved_or_stable"])
        self.assertEqual(trend["regression_count"], 0)


if __name__ == "__main__":
    unittest.main()
