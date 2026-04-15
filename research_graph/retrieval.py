from __future__ import annotations

import math
from collections import Counter

from . import storage
from .dedup import tokens


def _term_freq(text: str) -> dict[str, int]:
    return dict(Counter(tokens(text)))


def index_source(conn, source_id: int, content: str) -> None:
    storage.upsert_embedding(conn, "source", source_id, _term_freq(content))


def index_thinking(conn, thinking_id: int, content: str) -> None:
    storage.upsert_embedding(conn, "thinking", thinking_id, _term_freq(content))


def _score(query_terms: list[str], doc_terms: dict[str, int], idf: dict[str, float]) -> float:
    if not doc_terms:
        return 0.0
    doc_len = sum(doc_terms.values())
    score = 0.0
    for qt in set(query_terms):
        tf = doc_terms.get(qt, 0) / doc_len
        score += tf * idf.get(qt, 0.0)
    return score


def _rank(
    query: str, embeddings: list[dict], top_k: int, min_score: float
) -> list[tuple[int, float]]:
    qt = tokens(query)
    if not qt or not embeddings:
        return []
    n = len(embeddings)
    df: Counter = Counter()
    for e in embeddings:
        for term in e["terms"].keys():
            df[term] += 1
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}
    scored = [
        (e["object_id"], _score(qt, e["terms"], idf)) for e in embeddings
    ]
    scored = [(i, s) for i, s in scored if s > min_score]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]


def search_sources(
    conn,
    query: str,
    objective_id: int | None = None,
    topic_id: int | None = None,
    since: str | None = None,
    until: str | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    sql = "SELECT s.id FROM sources s JOIN objectives o ON o.id = s.objective_id WHERE 1=1"
    args: list = []
    if objective_id is not None:
        sql += " AND s.objective_id = ?"
        args.append(objective_id)
    if topic_id is not None:
        sql += " AND o.topic_id = ?"
        args.append(topic_id)
    if since is not None:
        sql += " AND s.fetched_at >= ?"
        args.append(since)
    if until is not None:
        sql += " AND s.fetched_at <= ?"
        args.append(until)
    ids = [int(r["id"]) for r in conn.execute(sql, args).fetchall()]
    embs = storage.get_embeddings(conn, "source", ids)
    ranked = _rank(query, embs, top_k, min_score)
    out = []
    for sid, score in ranked:
        row = storage.get_source(conn, sid)
        if row:
            row["score"] = score
            out.append(row)
    return out


def search_thinkings(
    conn,
    query: str,
    objective_id: int | None = None,
    topic_id: int | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    sql = "SELECT t.id FROM thinkings t JOIN objectives o ON o.id = t.objective_id WHERE 1=1"
    args: list = []
    if objective_id is not None:
        sql += " AND t.objective_id = ?"
        args.append(objective_id)
    if topic_id is not None:
        sql += " AND o.topic_id = ?"
        args.append(topic_id)
    ids = [int(r["id"]) for r in conn.execute(sql, args).fetchall()]
    embs = storage.get_embeddings(conn, "thinking", ids)
    ranked = _rank(query, embs, top_k, min_score)
    out = []
    for tid, score in ranked:
        row = storage.get_thinking(conn, tid)
        if row:
            row["score"] = score
            out.append(row)
    return out
