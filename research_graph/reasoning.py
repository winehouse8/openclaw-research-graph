"""Reasoning layer — actor proposes a Thinking, critic vets it.

⚠️ PLACEHOLDER IMPLEMENTATION ⚠️
================================

The current `actor_propose` is **NOT real reasoning**. It is a
keyword extractor: for each cited source it picks the top-N
non-stopword tokens that are not already in the objective question
and emits a `- source N | url | title keywords: foo, bar, baz` bullet.
The output is then trivially passed by the critic's verbatim-overlap
check because keywords are extracted from source text — never quoted
in long runs — by construction.

This is a deliberate placeholder so the orchestrator pipeline can be
exercised end-to-end without an LLM dependency. When real reasoning
is wired (LLM call via a pluggable actor backend mirroring
`ExternalSearch`), the critic's quote / citation / offset checks
become load-bearing. Until then:

- All `actor_propose` outputs carry the implicit `[placeholder]`
  semantic — do not treat the synthesis as a substantive answer.
- The critic exists to catch ONE failure mode (excessive verbatim
  quoting) and is meaningless against keyword output. It is wired up
  for the future LLM integration, not for the current actor.
- Spec L108-110 (`actor / critic 구조가 품질 향상에 도움`) is
  **structurally** satisfied (there are two roles, separated) but
  the quality lift will only land when actor_propose is replaced.

The orchestrator + retrieval + dedup + storage layers are spec-
compliant on their own; reasoning is the one piece that is honestly
a stub. See spec.md L108-110 ("구현 힌트, 비구속") — the spec
itself flags actor/critic as a hint not a hard requirement.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from . import store


# Set on every Thinking produced by the placeholder actor below so
# downstream code (and tests) can tell synthesis came from the
# placeholder vs a future LLM-backed actor without sniffing content.
ACTOR_BACKEND_PLACEHOLDER = "placeholder/keyword-extractor"


MAX_QUOTE_LEN = 200
MAX_QUOTES = 2
MIN_VERBATIM_RUN = 40  # chars; runs longer than this count as a quote


@dataclass
class CriticVerdict:
    accepted: bool
    reasons: list[str]


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "by", "is", "are", "was", "were", "be", "been", "as",
    "this", "that", "it", "its", "from", "into", "than", "then", "so",
}


_MAX_KEYWORDS_PER_SOURCE = 20


def actor_propose(sources: list[dict], objective_question: str) -> tuple[str, list[int]]:
    """Synthesize a thinking from sources by extracting distinct keywords."""
    if not sources:
        return ("No sources available to reason from.", [])
    bullets = []
    used: list[int] = []
    evidence_ids: list[str] = []
    for s in sources:
        toks = [
            t.lower()
            for t in re.findall(r"[A-Za-z0-9]+", s["content"])
            if t.lower() not in _STOPWORDS and len(t) > 1
        ]
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


def _verbatim_overlap(thinking: str, source_text: str) -> int:
    if not thinking or not source_text:
        return 0
    matcher = difflib.SequenceMatcher(a=source_text, b=thinking, autojunk=False)
    match = matcher.find_longest_match(0, len(source_text), 0, len(thinking))
    return int(match.size)


def critic_verify(backend, thinking_text: str, cited_source_ids: list[int]) -> CriticVerdict:
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
            reasons.append(f"source {s['id']} repeated verbatim {thinking_text.count(body)} times")
    if quote_run_count > MAX_QUOTES:
        reasons.append(f"too many verbatim quotes: {quote_run_count} > {MAX_QUOTES}")

    return CriticVerdict(len(reasons) == 0, reasons)
