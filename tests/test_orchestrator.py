"""Tests for the white-box closed-loop research orchestrator.

All tests run without Neo4j or any external services by using
stub implementations of Executor, Critic, and TaskGenerator.
"""

from research_graph.orchestrator import (
    CritiqueReport,
    CycleReport,
    ExecutionResult,
    FileReportSink,
    LogReportSink,
    LoopDecision,
    MultiSink,
    ResearchLoop,
    RuleCritic,
    TaskEnvelope,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Stub implementations for testing
# ---------------------------------------------------------------------------

class StubExecutor:
    """Returns canned results. Tracks calls for assertions."""

    def __init__(self, results=None):
        self.calls = []
        self._results = results or []
        self._idx = 0

    def execute(self, task: TaskEnvelope) -> ExecutionResult:
        self.calls.append(task)
        if self._idx < len(self._results):
            result = self._results[self._idx]
            self._idx += 1
            return result
        return ExecutionResult(
            task_id=task.task_id,
            status="success",
            summary="Default stub result",
            findings=["finding-1"],
            sources_found=[{"title": "src-1", "canonical_url": "https://example.com"}],
        )


class StubTaskGenerator:
    """Generates simple sequential tasks."""

    def generate(self, objective_id, objective_context, prior_critique, iteration):
        hint = ""
        if prior_critique and prior_critique.next_task_hint:
            hint = f" [{prior_critique.next_task_hint}]"
        return TaskEnvelope(
            task_id=f"task-{iteration}",
            objective_id=objective_id,
            iteration=iteration,
            description=f"Research iteration {iteration}{hint}",
            task_type="research",
            context_summary=f"Context for iteration {iteration}",
        )


class TerminateAfterNCritic:
    """Critiques: returns CONTINUE for N iterations, then TERMINATE_SUCCESS."""

    def __init__(self, n: int):
        self._n = n

    def critique(self, task, result, objective_context, iteration):
        if iteration >= self._n:
            decision = LoopDecision.TERMINATE_SUCCESS
            next_hint = None
        else:
            decision = LoopDecision.CONTINUE
            next_hint = f"Continue to iteration {iteration + 1}"
        return CritiqueReport(
            task_id=task.task_id,
            iteration=iteration,
            objective_id=task.objective_id,
            decision=decision,
            confidence=0.7,
            strengths=["completed"],
            weaknesses=[],
            gaps=[] if iteration >= self._n else ["more research needed"],
            next_task_hint=next_hint,
            reasoning=f"Iteration {iteration} of {self._n}",
        )


class RecordingSink:
    """Records all events for test assertions."""

    def __init__(self):
        self.dispatched = []
        self.executions = []
        self.critiques = []
        self.cycles = []
        self.loop_ends = []

    def on_task_dispatched(self, task):
        self.dispatched.append(task)

    def on_execution_complete(self, task, result):
        self.executions.append((task, result))

    def on_critique_complete(self, critique):
        self.critiques.append(critique)

    def on_cycle_complete(self, report):
        self.cycles.append(report)

    def on_loop_end(self, objective_id, reason, total_iterations):
        self.loop_ends.append((objective_id, reason, total_iterations))


# ---------------------------------------------------------------------------
# Tests: Core loop behavior
# ---------------------------------------------------------------------------

class TestResearchLoopBasic:
    def test_loop_runs_and_terminates(self):
        """Loop runs exactly N iterations then terminates on success."""
        sink = RecordingSink()
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=3),
            task_generator=StubTaskGenerator(),
            sink=sink,
            max_iterations=10,
        )

        result = loop.run("obj-001")

        assert result["status"] == "success"
        assert result["iterations"] == 3
        assert len(result["cycle_reports"]) == 3
        assert result["final_critique"]["decision"] == "success"

    def test_loop_respects_max_iterations(self):
        """Loop stops at max_iterations even if critic says continue."""
        sink = RecordingSink()
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=999),  # never terminates
            task_generator=StubTaskGenerator(),
            sink=sink,
            max_iterations=5,
        )

        result = loop.run("obj-002")

        assert result["status"] == "max_iterations"
        assert result["iterations"] == 5

    def test_single_iteration_terminate(self):
        """Loop can terminate after just one iteration."""
        sink = RecordingSink()
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=1),
            task_generator=StubTaskGenerator(),
            sink=sink,
        )

        result = loop.run("obj-003")

        assert result["status"] == "success"
        assert result["iterations"] == 1


class TestSinkNotifications:
    def test_all_sink_events_fire(self):
        """Every sink method is called in the correct order."""
        sink = RecordingSink()
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=2),
            task_generator=StubTaskGenerator(),
            sink=sink,
        )

        loop.run("obj-004")

        assert len(sink.dispatched) == 2
        assert len(sink.executions) == 2
        assert len(sink.critiques) == 2
        assert len(sink.cycles) == 2
        assert len(sink.loop_ends) == 1
        assert sink.loop_ends[0] == ("obj-004", "success", 2)

    def test_sink_receives_task_before_execution(self):
        """Task dispatch event fires before execution completes."""
        events = []

        class OrderTrackingSink:
            def on_task_dispatched(self, task):
                events.append(("dispatch", task.task_id))
            def on_execution_complete(self, task, result):
                events.append(("exec", task.task_id))
            def on_critique_complete(self, critique):
                events.append(("critique", critique.task_id))
            def on_cycle_complete(self, report):
                events.append(("cycle", report.iteration))
            def on_loop_end(self, objective_id, reason, total_iterations):
                events.append(("end", reason))

        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=1),
            task_generator=StubTaskGenerator(),
            sink=OrderTrackingSink(),
        )
        loop.run("obj-005")

        assert events == [
            ("dispatch", "task-1"),
            ("exec", "task-1"),
            ("critique", "task-1"),
            ("cycle", 1),
            ("end", "success"),
        ]


class TestCritiqueFlowsToNextTask:
    def test_critique_hint_reaches_next_task(self):
        """The critique's next_task_hint is available to the task generator."""
        generator = StubTaskGenerator()
        sink = RecordingSink()
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=3),
            task_generator=generator,
            sink=sink,
        )

        loop.run("obj-006")

        # Second task should include the hint from first critique
        second_task = sink.dispatched[1]
        assert "Continue to iteration 2" in second_task.description

    def test_prior_critique_passed_to_generator(self):
        """TaskGenerator receives the prior critique for decision-making."""
        received_critiques = []

        class TrackingGenerator:
            def generate(self, objective_id, objective_context, prior_critique, iteration):
                received_critiques.append(prior_critique)
                return TaskEnvelope(
                    task_id=f"task-{iteration}",
                    objective_id=objective_id,
                    iteration=iteration,
                    description=f"Task {iteration}",
                )

        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=2),
            task_generator=TrackingGenerator(),
            sink=RecordingSink(),
        )
        loop.run("obj-007")

        assert received_critiques[0] is None  # first iteration has no prior
        assert received_critiques[1] is not None
        assert received_critiques[1].decision == LoopDecision.CONTINUE


class TestExecutorFailure:
    def test_executor_exception_becomes_failure_result(self):
        """If executor raises, loop captures it and continues."""
        class ExplodingExecutor:
            def execute(self, task):
                if task.iteration == 1:
                    raise RuntimeError("connection refused")
                return ExecutionResult(
                    task_id=task.task_id,
                    status="success",
                    summary="recovered",
                    findings=["f1"],
                )

        class PivotOnFailureCritic:
            def critique(self, task, result, objective_context, iteration):
                if result.status == "failure":
                    return CritiqueReport(
                        task_id=task.task_id, iteration=iteration,
                        objective_id=task.objective_id,
                        decision=LoopDecision.PIVOT, confidence=0.3,
                        weaknesses=["executor failed"],
                        next_task_hint="retry with different approach",
                    )
                return CritiqueReport(
                    task_id=task.task_id, iteration=iteration,
                    objective_id=task.objective_id,
                    decision=LoopDecision.TERMINATE_SUCCESS, confidence=0.8,
                    strengths=["recovered"],
                )

        sink = RecordingSink()
        loop = ResearchLoop(
            executor=ExplodingExecutor(),
            critic=PivotOnFailureCritic(),
            task_generator=StubTaskGenerator(),
            sink=sink,
        )

        result = loop.run("obj-008")

        assert result["status"] == "success"
        assert result["iterations"] == 2
        # First cycle should show failure, second should show success
        assert result["cycle_reports"][0]["decision"] == "pivot"
        assert result["cycle_reports"][1]["decision"] == "success"

    def test_blocked_terminates(self):
        """TERMINATE_BLOCKED stops the loop immediately."""
        class BlockCritic:
            def critique(self, task, result, objective_context, iteration):
                return CritiqueReport(
                    task_id=task.task_id, iteration=iteration,
                    objective_id=task.objective_id,
                    decision=LoopDecision.TERMINATE_BLOCKED, confidence=0.9,
                    weaknesses=["no access to required resource"],
                )

        sink = RecordingSink()
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=BlockCritic(),
            task_generator=StubTaskGenerator(),
            sink=sink,
        )

        result = loop.run("obj-009")

        assert result["status"] == "blocked"
        assert result["iterations"] == 1


class TestRuleCritic:
    def test_success_with_findings_and_sources(self):
        """RuleCritic: success + findings + sources = good critique."""
        critic = RuleCritic()
        task = TaskEnvelope(task_id="t1", objective_id="o1", iteration=1, description="test")
        result = ExecutionResult(
            task_id="t1", status="success", summary="done",
            findings=["f1", "f2"],
            sources_found=[{"title": "s1"}],
            thinkings_produced=[{"title": "th1"}],
        )
        context = {
            "relevant_source_summaries": [{"node_id": "existing"}],
            "latest_thinking_summaries": [{"node_id": "existing"}],
        }

        critique = critic.critique(task, result, context, 1)

        assert critique.decision == LoopDecision.TERMINATE_SUCCESS
        assert critique.confidence > 0.5
        assert len(critique.strengths) >= 3

    def test_failure_causes_pivot(self):
        """RuleCritic: executor failure = PIVOT decision."""
        critic = RuleCritic()
        task = TaskEnvelope(task_id="t2", objective_id="o1", iteration=1, description="test")
        result = ExecutionResult(
            task_id="t2", status="failure", summary="failed",
            error="timeout",
        )

        critique = critic.critique(task, result, {}, 1)

        assert critique.decision == LoopDecision.PIVOT
        assert critique.confidence < 0.5

    def test_no_sources_creates_gap(self):
        """RuleCritic: no sources found = gap noted."""
        critic = RuleCritic()
        task = TaskEnvelope(task_id="t3", objective_id="o1", iteration=1, description="test")
        result = ExecutionResult(
            task_id="t3", status="success", summary="done",
            findings=["f1"],
        )

        critique = critic.critique(task, result, {}, 1)

        assert any("source" in g.lower() for g in critique.gaps)
        assert critique.decision == LoopDecision.CONTINUE


class TestCycleReport:
    def test_serialization(self):
        """CycleReport serializes to valid JSON."""
        report = CycleReport(
            iteration=1,
            objective_id="obj-001",
            task={"task_id": "t1", "description": "test"},
            execution_result={"status": "success", "summary": "done"},
            critique={"decision": "continue", "confidence": 0.7},
            decision="continue",
            next_task_hint="do more",
        )

        json_str = report.to_json()
        parsed = __import__("json").loads(json_str)

        assert parsed["iteration"] == 1
        assert parsed["decision"] == "continue"
        assert parsed["next_task_hint"] == "do more"


class TestMultiSink:
    def test_fans_out_to_all_sinks(self):
        """MultiSink dispatches to every child sink."""
        s1 = RecordingSink()
        s2 = RecordingSink()
        multi = MultiSink([s1, s2])

        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=1),
            task_generator=StubTaskGenerator(),
            sink=multi,
        )
        loop.run("obj-010")

        assert len(s1.dispatched) == 1
        assert len(s2.dispatched) == 1
        assert len(s1.loop_ends) == 1
        assert len(s2.loop_ends) == 1


class TestFileReportSink:
    def test_writes_files(self, tmp_path):
        """FileReportSink creates JSON files for each event."""
        sink = FileReportSink(str(tmp_path))
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=1),
            task_generator=StubTaskGenerator(),
            sink=sink,
        )
        loop.run("obj-011")

        files = list(tmp_path.glob("*.json"))
        names = {f.stem for f in files}

        # Should have task, result, critique, cycle, and loop-end files
        assert any("task-" in n for n in names)
        assert any("result-" in n for n in names)
        assert any("critique-" in n for n in names)
        assert any("cycle-" in n for n in names)
        assert "loop-end" in names


class TestContextProvider:
    def test_context_provider_called_each_iteration(self):
        """context_provider is called once per iteration with the objective_id."""
        calls = []

        def track_context(objective_id):
            calls.append(objective_id)
            return {"relevant_source_summaries": [], "latest_thinking_summaries": []}

        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=3),
            task_generator=StubTaskGenerator(),
            sink=RecordingSink(),
            context_provider=track_context,
        )
        loop.run("obj-012")

        assert calls == ["obj-012", "obj-012", "obj-012"]


class TestCycleReportsAccessible:
    def test_cycle_reports_property(self):
        """cycle_reports property returns all reports from the run."""
        loop = ResearchLoop(
            executor=StubExecutor(),
            critic=TerminateAfterNCritic(n=2),
            task_generator=StubTaskGenerator(),
            sink=RecordingSink(),
        )
        loop.run("obj-013")

        reports = loop.cycle_reports
        assert len(reports) == 2
        assert reports[0].iteration == 1
        assert reports[1].iteration == 2
