"""White-box closed-loop research orchestrator.

This module implements a visible, loggable research loop where:
1. A task is generated from the current research objective state
2. A child executor (Claude Code, OMC, or stub) runs the task
3. The result is ingested and critiqued against the objective
4. A next-task decision is made (continue, pivot, or terminate)
5. Every step is reported to external sinks (Discord, file log, etc.)

The loop never silently stops at child success — it always critiques
and decides whether to continue.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .utils import now_iso, stable_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TaskStatus(enum.Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class LoopDecision(enum.Enum):
    CONTINUE = "continue"          # generate next task, keep looping
    PIVOT = "pivot"                # change approach, keep looping
    TERMINATE_SUCCESS = "success"  # objective met
    TERMINATE_BLOCKED = "blocked"  # cannot proceed
    TERMINATE_MAX_ITER = "max_iterations"


@dataclass
class TaskEnvelope:
    """A research task dispatched to a child executor."""
    task_id: str
    objective_id: str
    iteration: int
    description: str
    task_type: str = "research"       # research | synthesis | verification
    context_summary: str = ""         # what the child should know
    parent_task_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ExecutionResult:
    """What the child executor returns."""
    task_id: str
    status: str                       # "success" | "failure" | "partial"
    summary: str                      # human-readable result summary
    findings: List[str] = field(default_factory=list)
    sources_found: List[Dict[str, Any]] = field(default_factory=list)
    thinkings_produced: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    raw_output: Optional[str] = None  # full child output for logging
    completed_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CritiqueReport:
    """Parent's critique of a child execution result."""
    task_id: str
    iteration: int
    objective_id: str
    decision: LoopDecision
    confidence: float                  # 0.0–1.0, how confident the critique is
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    next_task_hint: Optional[str] = None
    reasoning: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass
class CycleReport:
    """Complete report for one loop iteration — the white-box artifact."""
    iteration: int
    objective_id: str
    task: Dict[str, Any]
    execution_result: Dict[str, Any]
    critique: Dict[str, Any]
    decision: str
    next_task_hint: Optional[str]
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Protocols (pluggable seams)
# ---------------------------------------------------------------------------

@runtime_checkable
class Executor(Protocol):
    """Runs a research task and returns a result.

    Implementations:
    - ClaudeCodeExecutor: spawns a Claude Code child process
    - StubExecutor: for testing
    """

    def execute(self, task: TaskEnvelope) -> ExecutionResult: ...


@runtime_checkable
class Critic(Protocol):
    """Critiques an execution result against the objective state.

    Implementations:
    - LLMCritic: uses an LLM to evaluate quality
    - RuleCritic: simple rule-based evaluation for testing
    """

    def critique(
        self,
        task: TaskEnvelope,
        result: ExecutionResult,
        objective_context: Dict[str, Any],
        iteration: int,
    ) -> CritiqueReport: ...


@runtime_checkable
class TaskGenerator(Protocol):
    """Generates the next task based on critique and accumulated state.

    Implementations:
    - LLMTaskGenerator: uses an LLM to decide what to research next
    - StaticTaskGenerator: for testing
    """

    def generate(
        self,
        objective_id: str,
        objective_context: Dict[str, Any],
        prior_critique: Optional[CritiqueReport],
        iteration: int,
    ) -> TaskEnvelope: ...


@runtime_checkable
class ReportSink(Protocol):
    """Receives cycle reports for external visibility.

    Implementations:
    - DiscordReportSink: posts to a Discord channel
    - FileReportSink: writes JSON to a log directory
    - LogReportSink: writes to Python logging
    - MultiSink: fans out to multiple sinks
    """

    def on_task_dispatched(self, task: TaskEnvelope) -> None: ...
    def on_execution_complete(self, task: TaskEnvelope, result: ExecutionResult) -> None: ...
    def on_critique_complete(self, critique: CritiqueReport) -> None: ...
    def on_cycle_complete(self, report: CycleReport) -> None: ...
    def on_loop_end(self, objective_id: str, reason: str, total_iterations: int) -> None: ...


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------

class LogReportSink:
    """Logs all cycle events to Python logging — always-on default sink."""

    def on_task_dispatched(self, task: TaskEnvelope) -> None:
        logger.info(
            "[dispatch] iter=%d task=%s type=%s desc=%s",
            task.iteration, task.task_id, task.task_type, task.description[:120],
        )

    def on_execution_complete(self, task: TaskEnvelope, result: ExecutionResult) -> None:
        logger.info(
            "[result] task=%s status=%s summary=%s",
            task.task_id, result.status, result.summary[:200],
        )

    def on_critique_complete(self, critique: CritiqueReport) -> None:
        logger.info(
            "[critique] iter=%d decision=%s confidence=%.2f gaps=%d reasoning=%s",
            critique.iteration, critique.decision.value, critique.confidence,
            len(critique.gaps), critique.reasoning[:200],
        )

    def on_cycle_complete(self, report: CycleReport) -> None:
        logger.info(
            "[cycle] iter=%d decision=%s next_hint=%s",
            report.iteration, report.decision, report.next_task_hint,
        )

    def on_loop_end(self, objective_id: str, reason: str, total_iterations: int) -> None:
        logger.info(
            "[loop_end] objective=%s reason=%s iterations=%d",
            objective_id, reason, total_iterations,
        )


class FileReportSink:
    """Writes each cycle report as a JSON file for post-hoc inspection."""

    def __init__(self, output_dir: str):
        import os
        self._dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _write(self, name: str, data: Dict[str, Any]) -> None:
        from pathlib import Path
        path = Path(self._dir) / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def on_task_dispatched(self, task: TaskEnvelope) -> None:
        self._write(f"task-{task.iteration:03d}-{task.task_id[:12]}", task.to_dict())

    def on_execution_complete(self, task: TaskEnvelope, result: ExecutionResult) -> None:
        self._write(f"result-{task.iteration:03d}-{task.task_id[:12]}", result.to_dict())

    def on_critique_complete(self, critique: CritiqueReport) -> None:
        self._write(f"critique-{critique.iteration:03d}-{critique.task_id[:12]}", critique.to_dict())

    def on_cycle_complete(self, report: CycleReport) -> None:
        self._write(f"cycle-{report.iteration:03d}", report.to_dict())

    def on_loop_end(self, objective_id: str, reason: str, total_iterations: int) -> None:
        self._write("loop-end", {
            "objective_id": objective_id,
            "reason": reason,
            "total_iterations": total_iterations,
            "timestamp": now_iso(),
        })


class MultiSink:
    """Fans out to multiple ReportSinks."""

    def __init__(self, sinks: List[ReportSink]):
        self._sinks = sinks

    def on_task_dispatched(self, task: TaskEnvelope) -> None:
        for s in self._sinks:
            s.on_task_dispatched(task)

    def on_execution_complete(self, task: TaskEnvelope, result: ExecutionResult) -> None:
        for s in self._sinks:
            s.on_execution_complete(task, result)

    def on_critique_complete(self, critique: CritiqueReport) -> None:
        for s in self._sinks:
            s.on_critique_complete(critique)

    def on_cycle_complete(self, report: CycleReport) -> None:
        for s in self._sinks:
            s.on_cycle_complete(report)

    def on_loop_end(self, objective_id: str, reason: str, total_iterations: int) -> None:
        for s in self._sinks:
            s.on_loop_end(objective_id, reason, total_iterations)


# ---------------------------------------------------------------------------
# Rule-based critic (usable without LLM, good for testing)
# ---------------------------------------------------------------------------

class RuleCritic:
    """Simple rule-based critic for bootstrapping and testing.

    Evaluates based on structural signals:
    - Did the executor succeed?
    - Did it produce findings?
    - Did it find sources?
    - Are there still gaps from the objective context?
    """

    def critique(
        self,
        task: TaskEnvelope,
        result: ExecutionResult,
        objective_context: Dict[str, Any],
        iteration: int,
    ) -> CritiqueReport:
        strengths: List[str] = []
        weaknesses: List[str] = []
        gaps: List[str] = []

        # Evaluate execution status
        if result.status == "success":
            strengths.append("Task completed successfully")
        elif result.status == "partial":
            weaknesses.append("Task only partially completed")
        else:
            weaknesses.append(f"Task failed: {result.error or 'unknown error'}")

        # Evaluate findings
        if result.findings:
            strengths.append(f"Produced {len(result.findings)} findings")
        else:
            weaknesses.append("No findings produced")

        # Evaluate sources
        if result.sources_found:
            strengths.append(f"Found {len(result.sources_found)} sources")
        else:
            gaps.append("No new sources discovered")

        # Evaluate thinkings
        if result.thinkings_produced:
            strengths.append(f"Produced {len(result.thinkings_produced)} analysis nodes")

        # Check objective context for existing coverage
        existing_sources = objective_context.get("relevant_source_summaries", [])
        existing_thinkings = objective_context.get("latest_thinking_summaries", [])
        if not existing_sources and not result.sources_found:
            gaps.append("Objective has no source coverage at all")
        if not existing_thinkings and not result.thinkings_produced:
            gaps.append("Objective has no analytical coverage")

        # Check open questions
        open_questions = objective_context.get("open_questions", [])
        if open_questions:
            gaps.extend(open_questions[:2])

        # Decision logic
        if result.status == "failure":
            decision = LoopDecision.PIVOT
            confidence = 0.3
            next_hint = "Retry with a different approach or narrower scope"
        elif not gaps:
            decision = LoopDecision.TERMINATE_SUCCESS
            confidence = 0.8
            next_hint = None
        elif len(strengths) > len(weaknesses) and result.findings:
            decision = LoopDecision.CONTINUE
            confidence = 0.6
            next_hint = f"Address gap: {gaps[0]}" if gaps else None
        else:
            decision = LoopDecision.CONTINUE
            confidence = 0.4
            next_hint = f"Strengthen coverage: {gaps[0]}" if gaps else "Broaden search"

        reasoning = (
            f"Strengths: {'; '.join(strengths)}. "
            f"Weaknesses: {'; '.join(weaknesses)}. "
            f"Gaps: {'; '.join(gaps)}."
        )

        return CritiqueReport(
            task_id=task.task_id,
            iteration=iteration,
            objective_id=task.objective_id,
            decision=decision,
            confidence=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            gaps=gaps,
            next_task_hint=next_hint,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# The orchestration loop
# ---------------------------------------------------------------------------

class ResearchLoop:
    """White-box closed-loop research orchestrator.

    Each iteration:
    1. Generate a task from objective state + prior critique
    2. Dispatch task to executor (child Claude Code / OMC / stub)
    3. Ingest execution result
    4. Critique result against objective
    5. Report everything to sinks (Discord, file, log)
    6. Decide: continue / pivot / terminate

    The loop NEVER silently stops at child success. It always
    critiques, reports, and explicitly decides whether to continue.

    All state transitions are externally visible through ReportSink.
    """

    def __init__(
        self,
        executor: Executor,
        critic: Critic,
        task_generator: TaskGenerator,
        sink: ReportSink,
        max_iterations: int = 10,
        context_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self._executor = executor
        self._critic = critic
        self._generator = task_generator
        self._sink = sink
        self._max_iterations = max_iterations
        self._context_provider = context_provider or (lambda oid: {})
        self._cycle_reports: List[CycleReport] = []

    @property
    def cycle_reports(self) -> List[CycleReport]:
        """All cycle reports from the most recent run — for programmatic access."""
        return list(self._cycle_reports)

    def run(self, objective_id: str) -> Dict[str, Any]:
        """Run the closed loop until termination.

        Returns a final summary dict with:
        - status: the terminal LoopDecision value
        - iterations: number of iterations completed
        - cycle_reports: list of all cycle report dicts
        - final_critique: the last critique dict
        """
        self._cycle_reports = []
        prior_critique: Optional[CritiqueReport] = None

        for iteration in range(1, self._max_iterations + 1):
            # --- Step 1: Get current objective context ---
            objective_context = self._context_provider(objective_id)

            # --- Step 2: Generate task ---
            task = self._generator.generate(
                objective_id=objective_id,
                objective_context=objective_context,
                prior_critique=prior_critique,
                iteration=iteration,
            )
            task.status = TaskStatus.EXECUTING
            self._sink.on_task_dispatched(task)

            # --- Step 3: Execute task (child runs here) ---
            try:
                result = self._executor.execute(task)
            except Exception as exc:
                result = ExecutionResult(
                    task_id=task.task_id,
                    status="failure",
                    summary=f"Executor raised exception: {exc}",
                    error=str(exc),
                )

            task.status = TaskStatus.COMPLETED if result.status == "success" else TaskStatus.FAILED
            self._sink.on_execution_complete(task, result)

            # --- Step 4: Critique result ---
            critique = self._critic.critique(
                task=task,
                result=result,
                objective_context=objective_context,
                iteration=iteration,
            )
            self._sink.on_critique_complete(critique)
            prior_critique = critique

            # --- Step 5: Build cycle report (white-box artifact) ---
            cycle_report = CycleReport(
                iteration=iteration,
                objective_id=objective_id,
                task=task.to_dict(),
                execution_result=result.to_dict(),
                critique=critique.to_dict(),
                decision=critique.decision.value,
                next_task_hint=critique.next_task_hint,
            )
            self._cycle_reports.append(cycle_report)
            self._sink.on_cycle_complete(cycle_report)

            # --- Step 6: Decide whether to continue ---
            if critique.decision in (
                LoopDecision.TERMINATE_SUCCESS,
                LoopDecision.TERMINATE_BLOCKED,
            ):
                self._sink.on_loop_end(objective_id, critique.decision.value, iteration)
                return self._build_summary(critique.decision.value, iteration, critique)

        # Max iterations reached
        self._sink.on_loop_end(objective_id, LoopDecision.TERMINATE_MAX_ITER.value, self._max_iterations)
        final_critique = prior_critique
        return self._build_summary(
            LoopDecision.TERMINATE_MAX_ITER.value,
            self._max_iterations,
            final_critique,
        )

    def _build_summary(
        self,
        status: str,
        iterations: int,
        final_critique: Optional[CritiqueReport],
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "iterations": iterations,
            "cycle_reports": [r.to_dict() for r in self._cycle_reports],
            "final_critique": final_critique.to_dict() if final_critique else None,
        }
