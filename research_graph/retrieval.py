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
    query: str, embeddings: list[dict], top_k: int, min_score: float,
    *,
    idf_corpus: dict | None = None,
    candidate_filter: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Score candidate embeddings against `query` using TF-IDF cosine.

    Optimisation hooks (added in iter-3, spec L51 + L79):
      - `idf_corpus`: a precomputed `{"total_docs": N, "df": {term: int}}`
        from the global retrieval index. Skips per-query DF rebuild.
        Falls back to local DF computation if None.
      - `candidate_filter`: a set of doc_ids that are allowed to score.
        The caller uses the global postings table to compute the
        "docs that share at least one query token" set, so the score
        loop only touches that subset rather than every embedding in
        scope. Falls back to "score every embedding" if None.

    Both hooks default to the legacy O(N · T) behaviour so existing
    callers that don't pass them keep working.
    """
    qt = tokens(query)
    if not qt or not embeddings:
        return []
    query_tf = dict(Counter(qt))
    if idf_corpus is not None:
        n = max(int(idf_corpus.get("total_docs") or len(embeddings)), 1)
        corpus_df = idf_corpus.get("df") or {}
    else:
        n = len(embeddings)
        corpus_df = {}
        for e in embeddings:
            for term in e["terms"].keys():
                corpus_df[term] = corpus_df.get(term, 0) + 1
    idf: dict[str, float] = {}
    all_terms = set(corpus_df.keys()) | set(query_tf.keys())
    for term in all_terms:
        idf[term] = math.log((1 + n) / (1 + corpus_df.get(term, 0))) + 1.0

    if candidate_filter is not None:
        # Inverted-index path: only score embeddings whose object_id
        # appears in the precomputed candidate set (= docs sharing at
        # least one query token). Iterating the full embeddings list
        # but skipping non-candidates is O(N) bookkeeping with O(K)
        # actual scoring work where K = |candidate_filter|. For
        # selective queries K << N.
        scored = [
            (e["object_id"], _cosine_score(query_tf, e["terms"], idf))
            for e in embeddings
            if int(e["object_id"]) in candidate_filter
        ]
    else:
        scored = [
            (e["object_id"], _cosine_score(query_tf, e["terms"], idf))
            for e in embeddings
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


def _build_candidate_filter(
    backend, kind: str, query: str, scope_ids: set[int]
) -> set[int]:
    """Return the intersection of (a) docs that contain at least one
    query token, with (b) the scope-restricted candidate id set.

    Uses the global retrieval index `postings` table from
    `store._get_retrieval_index_slot(backend, kind)`. Falls back to
    "no filter" (= score everything) if the index is empty / unbuilt.
    """
    qt = set(tokens(query))
    if not qt:
        return scope_ids
    slot = store._ensure_retrieval_index_built(backend, kind)
    postings = slot.get("postings") or {}
    if not postings:
        return scope_ids
    matched: set[int] = set()
    for tok in qt:
        bucket = postings.get(tok)
        if bucket:
            matched.update(bucket)
    return matched & scope_ids


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
    # Hot path optimisations (iter-3):
    #   1. Use the global IDF cache from the retrieval index so we
    #      don't recompute DF on every query.
    #   2. Prune the candidate set via the postings table — only docs
    #      that share at least one query token get scored.
    idf_corpus = store._ensure_retrieval_index_built(backend, "source")
    candidate_filter = _build_candidate_filter(
        backend, "source", query, set(ids)
    )
    ranked = _rank(
        query, embs, top_k, min_score,
        idf_corpus=idf_corpus,
        candidate_filter=candidate_filter,
    )
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
    idf_corpus = store._ensure_retrieval_index_built(backend, "thinking")
    candidate_filter = _build_candidate_filter(
        backend, "thinking", query, set(ids)
    )
    ranked = _rank(
        query, embs, top_k, min_score,
        idf_corpus=idf_corpus,
        candidate_filter=candidate_filter,
    )
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
