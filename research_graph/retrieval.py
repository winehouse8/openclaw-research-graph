from __future__ import annotations

import math
from collections import Counter

from . import store
from .dedup import tokens


def _term_freq(text: str) -> dict[str, int]:
    return dict(Counter(tokens(text)))


def index_source(backend, source_id: int, content: str) -> None:
    store.upsert_embedding(backend, "source", source_id, _term_freq(content))


def index_thinking(backend, thinking_id: int, content: str) -> None:
    store.upsert_embedding(backend, "thinking", thinking_id, _term_freq(content))


def _cosine_score(
    query_tf: dict[str, int],
    doc_terms: dict[str, int],
    idf: dict[str, float],
) -> float:
    if not doc_terms or not query_tf:
        return 0.0
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


def _candidate_sources(
    backend,
    objective_id: int | None,
    topic_id: int | None,
    since: str | None,
    until: str | None,
) -> list[dict]:
    if objective_id is not None:
        rows = store.list_sources(backend, objective_id)
    elif topic_id is not None:
        rows = []
        for obj in store.list_objectives(backend, topic_id):
            rows.extend(store.list_sources(backend, int(obj["id"])))
    else:
        rows = []
        for obj in store.list_objectives(backend):
            rows.extend(store.list_sources(backend, int(obj["id"])))
    if since is not None:
        rows = [r for r in rows if (r.get("fetched_at") or "") >= since]
    if until is not None:
        rows = [r for r in rows if (r.get("fetched_at") or "") <= until]
    return rows


def _candidate_thinkings(
    backend, objective_id: int | None, topic_id: int | None
) -> list[dict]:
    if objective_id is not None:
        return store.list_thinkings(backend, objective_id)
    if topic_id is not None:
        rows = []
        for obj in store.list_objectives(backend, topic_id):
            rows.extend(store.list_thinkings(backend, int(obj["id"])))
        return rows
    rows = []
    for obj in store.list_objectives(backend):
        rows.extend(store.list_thinkings(backend, int(obj["id"])))
    return rows


def search_sources(
    backend,
    query: str,
    objective_id: int | None = None,
    topic_id: int | None = None,
    since: str | None = None,
    until: str | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    candidates = _candidate_sources(backend, objective_id, topic_id, since, until)
    ids = [int(r["id"]) for r in candidates]
    embs = store.get_embeddings(backend, "source", ids)
    ranked = _rank(query, embs, top_k, min_score)
    by_id = {int(r["id"]): r for r in candidates}
    out: list[dict] = []
    for sid, score in ranked:
        row = by_id.get(int(sid))
        if row is None:
            continue
        row = dict(row)
        row.pop("_terms", None)
        row["score"] = score
        out.append(row)
    return out


def search_thinkings(
    backend,
    query: str,
    objective_id: int | None = None,
    topic_id: int | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    candidates = _candidate_thinkings(backend, objective_id, topic_id)
    ids = [int(r["id"]) for r in candidates]
    embs = store.get_embeddings(backend, "thinking", ids)
    ranked = _rank(query, embs, top_k, min_score)
    by_id = {int(r["id"]): r for r in candidates}
    out: list[dict] = []
    for tid, score in ranked:
        row = by_id.get(int(tid))
        if row is None:
            continue
        row = dict(row)
        row.pop("_terms", None)
        row["score"] = score
        out.append(row)
    return out
