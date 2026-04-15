from __future__ import annotations

import re
from dataclasses import dataclass

from . import storage


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


def actor_propose(sources: list[dict], objective_question: str) -> tuple[str, list[int]]:
    """Synthesize a thinking from sources by extracting distinct keywords -- never verbatim chunks."""
    if not sources:
        return ("No sources available to reason from.", [])
    bullets = []
    used: list[int] = []
    for s in sources:
        toks = [
            t.lower()
            for t in re.findall(r"[A-Za-z0-9]+", s["content"])
            if t.lower() not in _STOPWORDS and len(t) > 1
        ]
        seen = []
        for t in toks:
            if t not in seen:
                seen.append(t)
            if len(seen) >= 8:
                break
        bullets.append(f"- source {s['id']} keywords: " + ", ".join(seen))
        used.append(int(s["id"]))
    header = f"Synthesis for: {objective_question}"
    body = "\n".join(bullets)
    return (f"{header}\n{body}", used)


def _verbatim_overlap(thinking: str, source_text: str) -> int:
    """Length of longest verbatim run from source found in thinking."""
    if not thinking or not source_text:
        return 0
    best = 0
    n = len(source_text)
    step = 20
    for start in range(0, n - MIN_VERBATIM_RUN + 1, step):
        chunk = source_text[start : start + MIN_VERBATIM_RUN]
        if chunk in thinking:
            run = MIN_VERBATIM_RUN
            while start + run < n and thinking.find(source_text[start : start + run + 1]) != -1:
                run += 1
            best = max(best, run)
    return best


def critic_verify(conn, thinking_text: str, cited_source_ids: list[int]) -> CriticVerdict:
    reasons: list[str] = []
    sources: list[dict] = []
    for sid in cited_source_ids:
        s = storage.get_source(conn, sid)
        if not s:
            reasons.append(f"cited source {sid} does not exist")
            continue
        sources.append(s)
    if reasons:
        return CriticVerdict(False, reasons)

    quote_count = 0
    for s in sources:
        run = _verbatim_overlap(thinking_text, s["content"])
        if run > MAX_QUOTE_LEN:
            reasons.append(
                f"source {s['id']} quoted verbatim for {run} chars (max {MAX_QUOTE_LEN})"
            )
        elif run >= MIN_VERBATIM_RUN:
            quote_count += 1
        # also reject if the source body is repeated multiple times in the thinking
        body = s["content"].strip()
        if len(body) >= MIN_VERBATIM_RUN and thinking_text.count(body) >= 2:
            reasons.append(f"source {s['id']} repeated verbatim {thinking_text.count(body)} times")
    if quote_count > MAX_QUOTES:
        reasons.append(f"too many verbatim quotes: {quote_count} > {MAX_QUOTES}")

    return CriticVerdict(len(reasons) == 0, reasons)
