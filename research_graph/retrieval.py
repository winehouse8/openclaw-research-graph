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


def _cosine_score(
    query_tf: dict[str, int],
    doc_terms: dict[str, int],
    idf: dict[str, float],
) -> float:
    """Length-normalized cosine similarity over TF-IDF vectors.

    Dividing by the L2 norms of both the query and the document vector
    penalizes long documents that merely happen to contain every query
    token. A short document that matches a rare query term wins over a
    longer document that matches only common terms, which is the
    behaviour the e2e keyword-ranking tests exercise.
    """
    if not doc_terms or not query_tf:
        return 0.0
    # Query vector uses idf^2 weighting (standard tf-idf cosine form).
    dot = 0.0
    for qt, qf in query_tf.items():
        w_q = float(qf) * idf.get(qt, 0.0)
        w_d = float(doc_terms.get(qt, 0)) * idf.get(qt, 0.0)
        dot += w_q * w_d
    if dot == 0.0:
        return 0.0
    q_norm = math.sqrt(
        sum((float(qf) * idf.get(qt, 0.0)) ** 2 for qt, qf in query_tf.items())
    )
    d_norm = math.sqrt(
        sum((float(df) * idf.get(dt, 0.0)) ** 2 for dt, df in doc_terms.items())
    )
    if q_norm == 0.0 or d_norm == 0.0:
        return 0.0
    return dot / (q_norm * d_norm)


def _rank(
    query: str, embeddings: list[dict], top_k: int, min_score: float
) -> list[tuple[int, float]]:
    qt = tokens(query)
    if not qt or not embeddings:
        return []
    query_tf = dict(Counter(qt))
    n = len(embeddings)
    df: Counter = Counter()
    for e in embeddings:
        for term in e["terms"].keys():
            df[term] += 1
    # Smoothed idf; terms unseen in the corpus get idf 0 so they do not
    # inflate the cosine norms with empty dimensions.
    idf: dict[str, float] = {}
    all_terms = set(df.keys()) | set(query_tf.keys())
    for term in all_terms:
        idf[term] = math.log((1 + n) / (1 + df.get(term, 0))) + 1.0
    scored = [
        (e["object_id"], _cosine_score(query_tf, e["terms"], idf)) for e in embeddings
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
