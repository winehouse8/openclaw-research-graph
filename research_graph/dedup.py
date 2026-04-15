from __future__ import annotations

import hashlib
import re
from typing import Iterable

from . import storage


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_near_duplicate_source(
    conn, objective_id: int, content: str, threshold: float = 0.85
) -> dict | None:
    target = tokens(content)
    if not target:
        return None
    for row in storage.list_sources(conn, objective_id):
        if jaccard(target, tokens(row["content"])) >= threshold:
            return row
    return None


def find_near_duplicate_thinking(
    conn, objective_id: int, content: str, threshold: float = 0.85
) -> dict | None:
    target = tokens(content)
    if not target:
        return None
    for row in storage.list_thinkings(conn, objective_id):
        if jaccard(target, tokens(row["content"])) >= threshold:
            return row
    return None


def upsert_source(
    conn,
    objective_id: int,
    url: str | None,
    title: str | None,
    content: str,
    search_query: str | None = None,
    threshold: float = 0.85,
) -> tuple[int, bool]:
    """Insert a source with dedup. Returns (source_id, created)."""
    h = content_hash(content)
    existing = storage.find_source_by_hash(conn, h)
    if existing and existing["objective_id"] == objective_id:
        return int(existing["id"]), False
    near = find_near_duplicate_source(conn, objective_id, content, threshold)
    if near:
        return int(near["id"]), False
    sid = storage.insert_source(conn, objective_id, url, title, content, h, search_query)
    from . import retrieval

    retrieval.index_source(conn, sid, content)
    return sid, True


def upsert_thinking(
    conn,
    objective_id: int,
    content: str,
    supports_source_ids: list[int],
    author: str,
    supersedes_id: int | None = None,
    threshold: float = 0.85,
) -> tuple[int, bool]:
    h = content_hash(content)
    existing = storage.find_thinking_by_hash(conn, h)
    if existing and existing["objective_id"] == objective_id:
        return int(existing["id"]), False
    near = find_near_duplicate_thinking(conn, objective_id, content, threshold)
    if near:
        return int(near["id"]), False
    tid = storage.insert_thinking(
        conn, objective_id, content, h, supports_source_ids, author, supersedes_id
    )
    from . import retrieval

    retrieval.index_thinking(conn, tid, content)
    return tid, True
