"""Reasoning layer — actor proposes a Thinking, critic vets it.

Structural upgrade (iter-4, continual-research scaling):
  The legacy `actor_propose` is a keyword extractor — it cannot be
  improved by "adding more reasoning power" because there is nowhere
  to plug reasoning power in. This module now exposes a pluggable
  `ActorBackend` Protocol (mirroring `research_graph.search.ExternalSearch`)
  plus a quality-scoring helper so that:

    1. An LLM-backed actor (or any other implementation) can be
       registered at runtime without patching the orchestrator.
    2. Every Thinking carries a quantitative quality score that the
       orchestrator uses to gate supersession. A regression cannot
       silently overwrite a better prior answer — it becomes a
       sibling branch instead (hypothesis branching).
    3. The legacy keyword extractor still works as
       `PlaceholderActorBackend` for backwards compatibility; the
       free-function `actor_propose` is kept as a thin shim.

  Why quality matters for continual research: supersession used to
  trigger purely on "did we fetch new external sources?" — which made
  day-2 always overwrite day-1 regardless of reasoning improvement.
  With a score-gated supersession, the daily cron can only move the
  live tip when the evidence is actually better, giving the system a
  feedback signal that benefits from more reasoning horsepower.

Critic note: the verbatim-overlap check was designed to catch a FUTURE
LLM actor copy-pasting source text. It is trivially satisfied by the
placeholder keyword output (by construction), so for the placeholder
backend the critic is essentially a no-op. It becomes load-bearing
once an LLM backend is wired. spec.md L108-110 (actor/critic as a
non-binding hint) continues to hold.
"""
from __future__ import annotations

import difflib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import store


# ---------------------------------------------------------------------------
# Critic verbatim-overlap limits (unchanged; meaningful once an LLM
# actor is wired in).
# ---------------------------------------------------------------------------

MAX_QUOTE_LEN = 200
MAX_QUOTES = 2
MIN_VERBATIM_RUN = 40  # chars; runs longer than this count as a quote


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "by", "is", "are", "was", "were", "be", "been", "as",
    "this", "that", "it", "its", "from", "into", "than", "then", "so",
}

_MAX_KEYWORDS_PER_SOURCE = 20

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens_no_stop(text: str) -> list[str]:
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if t.lower() not in _STOPWORDS and len(t) > 1
    ]


# ---------------------------------------------------------------------------
# Pluggable ActorBackend — mirror of search.ExternalSearch.
# ---------------------------------------------------------------------------


class ActorBackend(Protocol):
    """Implementations take (sources, objective_question) and return
    (thinking_text, cited_source_ids). The `name` attribute is a stable
    identifier persisted on the Thinking row so provenance survives
    across backend swaps."""

    name: str

    def propose(
        self, sources: list[dict], objective_question: str
    ) -> tuple[str, list[int]]: ...


class PlaceholderActorBackend:
    """The legacy keyword-extractor, wrapped as an ActorBackend so the
    orchestrator can uniformly treat it like any other implementation.

    Output shape is unchanged from the pre-iter-4 `actor_propose` free
    function so existing tests that pin the exact text keep passing.
    """

    name = "placeholder/keyword-extractor"

    def propose(
        self, sources: list[dict], objective_question: str
    ) -> tuple[str, list[int]]:
        if not sources:
            return ("No sources available to reason from.", [])
        bullets: list[str] = []
        used: list[int] = []
        evidence_ids: list[str] = []
        for s in sources:
            toks = _tokens_no_stop(s.get("content", ""))
            seen: list[str] = []
            for t in toks:
                if t not in seen:
                    seen.append(t)
                if len(seen) >= _MAX_KEYWORDS_PER_SOURCE:
                    break
            url = (s.get("url") or "").strip()
            title = (s.get("title") or "").strip()
            header_bits = [f"source {s['id']}"]
            if url:
                header_bits.append(url)
            if title:
                header_bits.append(title)
            prefix = " | ".join(header_bits)
            bullets.append(f"- {prefix} keywords: " + ", ".join(seen))
            used.append(int(s["id"]))
            evidence_ids.append(str(s["id"]))
        header = f"Synthesis for: {objective_question}"
        evidence_line = f"Evidence set: [{', '.join(evidence_ids)}]"
        body = "\n".join(bullets)
        return (f"{header}\n{evidence_line}\n{body}", used)


# Legacy constant — tests and external callers may still import this.
ACTOR_BACKEND_PLACEHOLDER = PlaceholderActorBackend.name


_BUILTIN_ACTOR_BACKENDS: dict[str, Callable[[], ActorBackend]] = {
    "placeholder": PlaceholderActorBackend,
}


def register_actor_backend(name: str, factory: Callable[[], ActorBackend]) -> None:
    """Explicit extension point — trusted callers (tests, LLM adapters)
    install a backend factory under a short name. Mirrors
    `search.register_backend`. No import-time auto-discovery; OpenClaw
    research runs with an explicit allowlist by design."""
    _BUILTIN_ACTOR_BACKENDS[name] = factory


def get_default_actor_backend() -> ActorBackend:
    """Resolve the active backend via `OPENCLAW_RESEARCH_ACTOR` env var,
    defaulting to `placeholder`. Unknown names raise so a typo in a
    production cron doesn't silently fall through to the stub."""
    name = os.environ.get("OPENCLAW_RESEARCH_ACTOR", "placeholder").strip() or "placeholder"
    if name not in _BUILTIN_ACTOR_BACKENDS:
        raise ValueError(
            f"unknown actor backend {name!r}; allowed: {sorted(_BUILTIN_ACTOR_BACKENDS)}"
        )
    return _BUILTIN_ACTOR_BACKENDS[name]()


def actor_propose(
    sources: list[dict], objective_question: str
) -> tuple[str, list[int]]:
    """Back-compat shim for pre-iter-4 callers. Always uses the
    placeholder backend so existing tests that pin exact keyword output
    keep passing. New code should instantiate an `ActorBackend` and
    call `.propose(...)` directly, or pass one to `Orchestrator`."""
    return PlaceholderActorBackend().propose(sources, objective_question)


# ---------------------------------------------------------------------------
# Quality scoring — the structural feedback signal that makes more
# reasoning power actually matter.
# ---------------------------------------------------------------------------


@dataclass
class QualityScore:
    """A quantitative, deterministic, backend-agnostic quality score
    for a single Thinking.

    We deliberately do NOT try to score "truth" — there is no ground
    truth in continual research. Instead we score three evidence-
    hygiene signals that an LLM actor CAN improve on (and that the
    placeholder keyword extractor cannot fake):

      grounding: of the sources this Thinking cites, what fraction
                 actually share non-stopword tokens with the Thinking
                 text. A Thinking that cites source 42 without ever
                 referencing its vocabulary is ungrounded.
      coverage:  of the scope the retrieval layer considered (the
                 top-k ranked sources handed to the actor), what
                 fraction got cited. A Thinking that picks one source
                 and ignores four equally-ranked ones has low coverage.
      diversity: distinct URLs / citations. A Thinking that cites the
                 same URL five times has low diversity. A Thinking
                 that cites five distinct URLs has diversity 1.0.

    `overall` is a fixed weighted blend (not user-tunable by design —
    tuning belongs in a higher-level evaluator). All four fields live
    in [0, 1]. Empty inputs resolve to 0.0 (not NaN) so the field is
    always safely comparable.
    """

    grounding: float = 0.0
    coverage: float = 0.0
    diversity: float = 0.0
    overall: float = 0.0
    cited_count: int = 0
    scope_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "grounding": round(float(self.grounding), 4),
            "coverage": round(float(self.coverage), 4),
            "diversity": round(float(self.diversity), 4),
            "overall": round(float(self.overall), 4),
            "cited_count": int(self.cited_count),
            "scope_count": int(self.scope_count),
            "notes": list(self.notes),
        }


_GROUNDING_WEIGHT = 0.5
_COVERAGE_WEIGHT = 0.3
_DIVERSITY_WEIGHT = 0.2


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def score_thinking(
    thinking_text: str,
    cited_sources: list[dict],
    scope_sources: list[dict] | None = None,
) -> QualityScore:
    """Compute a QualityScore. Pure function, no backend dependency.

    Args:
      thinking_text: the full Thinking body (as the actor produced it).
      cited_sources: the dict rows for sources the actor actually cited
                     (NOT the ids — we need the content to measure
                     grounding).
      scope_sources: the dict rows for the retrieval-layer candidate
                     set that was handed to the actor (defaults to
                     `cited_sources` when unknown, which gives coverage
                     = 1.0 and isolates grounding).
    """
    notes: list[str] = []
    if not cited_sources:
        return QualityScore(notes=["no citations"])

    thinking_tokens = set(_tokens_no_stop(thinking_text))
    if not thinking_tokens:
        return QualityScore(
            cited_count=len(cited_sources),
            scope_count=len(scope_sources) if scope_sources else len(cited_sources),
            notes=["thinking text has no scoreable tokens"],
        )

    # grounding: fraction of citations that share >= 1 non-stopword
    # token with the thinking body.
    grounded = 0
    for s in cited_sources:
        src_tokens = set(_tokens_no_stop(s.get("content", "")))
        if src_tokens & thinking_tokens:
            grounded += 1
    grounding = grounded / len(cited_sources)

    # coverage: distinct cited IDs / distinct scope IDs. Degenerates
    # to 1.0 when scope is missing or trivially equals citations.
    cited_ids = {int(s["id"]) for s in cited_sources}
    if scope_sources:
        scope_ids = {int(s["id"]) for s in scope_sources}
        # Scope must be a superset of citations in practice (the actor
        # cites from the retrieved set). Defensive: guard against empty
        # scope edge case.
        denom = max(1, len(scope_ids))
        coverage = len(cited_ids & scope_ids) / denom
    else:
        coverage = 1.0
        notes.append("no scope provided; coverage defaulted to 1.0")

    # diversity: distinct citation URLs / number of citations. If the
    # actor cites source 5 three times the raw citation list has 3
    # entries but diversity drops.
    urls = [s.get("url") or f"__no_url__/{int(s['id'])}" for s in cited_sources]
    diversity = len(set(urls)) / max(1, len(urls))

    overall = _clamp01(
        _GROUNDING_WEIGHT * grounding
        + _COVERAGE_WEIGHT * coverage
        + _DIVERSITY_WEIGHT * diversity
    )
    return QualityScore(
        grounding=_clamp01(grounding),
        coverage=_clamp01(coverage),
        diversity=_clamp01(diversity),
        overall=overall,
        cited_count=len(cited_sources),
        scope_count=len(scope_sources) if scope_sources else len(cited_sources),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Critic (unchanged from iter-3 — verbatim-overlap watchdog).
# ---------------------------------------------------------------------------


@dataclass
class CriticVerdict:
    accepted: bool
    reasons: list[str]


def _verbatim_overlap(thinking: str, source_text: str) -> int:
    if not thinking or not source_text:
        return 0
    matcher = difflib.SequenceMatcher(a=source_text, b=thinking, autojunk=False)
    match = matcher.find_longest_match(0, len(source_text), 0, len(thinking))
    return int(match.size)


def critic_verify(
    backend, thinking_text: str, cited_source_ids: list[int]
) -> CriticVerdict:
    reasons: list[str] = []
    sources: list[dict] = []
    for sid in cited_source_ids:
        s = store.get_source(backend, sid)
        if not s:
            reasons.append(f"cited source {sid} does not exist")
            continue
        sources.append(s)
    if reasons:
        return CriticVerdict(False, reasons)

    max_run_chars = 0
    quote_run_count = 0
    for s in sources:
        run = _verbatim_overlap(thinking_text, s["content"])
        if run > max_run_chars:
            max_run_chars = run
        if run > MAX_QUOTE_LEN:
            reasons.append(
                f"source {s['id']} quoted verbatim for {run} chars (max {MAX_QUOTE_LEN})"
            )
        if run >= MIN_VERBATIM_RUN:
            quote_run_count += 1
        body = s["content"].strip()
        if len(body) >= MIN_VERBATIM_RUN and thinking_text.count(body) >= 2:
            reasons.append(
                f"source {s['id']} repeated verbatim {thinking_text.count(body)} times"
            )
    if quote_run_count > MAX_QUOTES:
        reasons.append(
            f"too many verbatim quotes: {quote_run_count} > {MAX_QUOTES}"
        )

    return CriticVerdict(len(reasons) == 0, reasons)
