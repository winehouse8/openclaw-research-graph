from __future__ import annotations

import hashlib
import re
from typing import Iterable

from . import store


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


def _row_token_set(row: dict) -> set[str]:
    """Get the token SET for an existing source/thinking row without
    re-tokenizing its content.

    The retrieval index already stores the token frequencies on the
    node as `_terms` (a dict[term -> tf]). Its keys are exactly the
    unique-token set we need for Jaccard. Read it directly.

    Fallback: if `_terms` is missing (legacy rows / non-dedup insert
    paths), fall back to live tokenization once.
    """
    cached = row.get("_terms")
    if cached:
        return set(cached.keys())
    return set(tokens(row.get("content", "") or ""))


# ---------------------------------------------------------------------------
# Inverted-index dedup (FIX B for the O(N²) per-objective dedup hot path)
# ---------------------------------------------------------------------------
#
# Background:
#   Before this iteration the dedup hot path was:
#       for row in store.list_sources(backend, objective_id):
#           jaccard(target, row_tokens(row))
#   which is O(N) per insert and therefore O(N²) over N inserts. A
#   measured baseline on the in-memory backend showed:
#       N=  100:    10 ms total (0.10 ms/insert)
#       N=  200:    37 ms       (0.19 ms/insert)
#       N=  400:   147 ms       (0.37 ms/insert)
#       N=  800:   587 ms       (0.73 ms/insert)
#       N= 1600:  2351 ms       (1.47 ms/insert)
#   per-insert cost grows linearly with N -> total is quadratic. Spec
#   L79 (long-running 구조 안정성) + L91 (daily 9am cron) make this
#   the dominant tail risk for the system.
#
# Strategy:
#   Maintain a per-(kind, objective_id) inverted index as a sidecar
#   attribute on the backend instance:
#       backend._dedup_index[(kind, objective_id)] = {
#           "postings": dict[token -> set[doc_id]],
#           "doc_token_count": dict[doc_id -> int],
#           "built": bool,
#       }
#   - LAZY BUILD: first call to `find_near_duplicate_*` for a given
#     (kind, oid) walks the existing rows once (O(N · avgT)) to build
#     the inverted index. Subsequent calls reuse it.
#   - INCREMENTAL UPDATE: after a successful `upsert_source/thinking`
#     insert, the new doc's tokens are added to postings + doc count.
#     No need to rebuild.
#   - STALE-TOLERANT: deletes that bypass dedup (e.g. cascade in
#     store.delete_objective) don't get a chance to update the index.
#     The lookup path verifies the doc still exists via
#     `store.get_source/get_thinking` before scoring; ghost entries
#     are skipped and lazily evicted from postings.
#   - SCOPED: indexes are per (kind, objective_id) so cross-objective
#     scopes never collide.
#
# Complexity:
#   - Build (one-shot per oid): O(N · avgT)  ← same as old per-call cost
#   - Insert (steady state):    O(target_terms)  ← O(1) amortized for fixed-length docs
#   - Find near-dup query:      O(target_terms · avg_postings) + O(K) score, where
#                               K = unique candidate ids returned by postings union.
#                               For sparse content K << N → typically O(target_terms).
#
# This is a per-process cache. Each new backend instance rebuilds
# from disk on first access. The spec's daily cron use case = single
# backend instance per cron run, so the rebuild cost is paid once
# per day across many objectives instead of N² times.

_INDEX_ATTR = "_dedup_index"


def _get_index_slot(backend, kind: str, objective_id: int) -> dict:
    indexes = getattr(backend, _INDEX_ATTR, None)
    if indexes is None:
        indexes = {}
        setattr(backend, _INDEX_ATTR, indexes)
    key = (kind, int(objective_id))
    slot = indexes.get(key)
    if slot is None:
        slot = {"postings": {}, "doc_token_count": {}, "built": False}
        indexes[key] = slot
    return slot


def _ensure_index_built(backend, kind: str, objective_id: int) -> dict:
    slot = _get_index_slot(backend, kind, objective_id)
    if slot["built"]:
        return slot
    # Lazy build: walk the existing rows once.
    if kind == "source":
        rows = store.list_sources(backend, objective_id)
    else:
        rows = store.list_thinkings(backend, objective_id)
    for row in rows:
        doc_id = int(row["id"])
        token_set = _row_token_set(row)
        if not token_set:
            continue
        slot["doc_token_count"][doc_id] = len(token_set)
        for tok in token_set:
            slot["postings"].setdefault(tok, set()).add(doc_id)
    slot["built"] = True
    return slot


def _register_in_index(
    backend, kind: str, objective_id: int, doc_id: int, token_set: set[str]
) -> None:
    if not token_set:
        return
    slot = _ensure_index_built(backend, kind, objective_id)
    slot["doc_token_count"][int(doc_id)] = len(token_set)
    for tok in token_set:
        slot["postings"].setdefault(tok, set()).add(int(doc_id))


def _find_near_duplicate_via_index(
    backend,
    kind: str,
    objective_id: int,
    target_tokens: set[str],
    threshold: float,
    list_fn,
    get_fn,
) -> dict | None:
    """O(target_terms · avg_postings) candidate lookup + O(K) Jaccard scoring."""
    if not target_tokens:
        return None
    slot = _ensure_index_built(backend, kind, objective_id)
    postings = slot["postings"]
    # Selective-candidate optimisation: rather than unioning postings
    # across ALL query tokens (which is O(N) when even one token is
    # common), we compute the intersection-count only over the K
    # SMALLEST postings lists. Documents that match the threshold
    # MUST share at least `inter_floor` tokens with the target, so
    # they MUST appear in at least one of the smallest postings (by
    # pigeonhole). Picking the smallest postings keeps the inner-loop
    # work proportional to the rarest tokens rather than the
    # commonest, which is the asymptotic win on diverse vocabularies.
    target_size = len(target_tokens)
    # Inter floor — minimum intersection count for any candidate to
    # possibly hit Jaccard >= threshold (since union >= target_size).
    from math import ceil
    inter_floor = ceil(threshold * target_size)
    # Gather (token, posting_size) for query tokens that exist in the
    # index, sort by posting size ascending. We only need to scan
    # enough small postings to guarantee any candidate above the
    # floor would appear in at least one of them. Pigeonhole: if a
    # candidate shares >= inter_floor tokens with the target, then it
    # must appear in at least (inter_floor) of those token postings.
    # If we walk the smallest (target_size - inter_floor + 1) postings,
    # we are guaranteed to see every candidate that meets the floor.
    sized_tokens: list[tuple[str, int]] = []
    for tok in target_tokens:
        bucket = postings.get(tok)
        if bucket:
            sized_tokens.append((tok, len(bucket)))
    sized_tokens.sort(key=lambda kv: kv[1])
    # Number of postings we MUST scan to be sound (pigeonhole):
    must_scan = max(1, target_size - inter_floor + 1)
    walk = sized_tokens[:must_scan]
    intersection_count: dict[int, int] = {}
    seed_ids: set[int] = set()
    for tok, _ in walk:
        bucket = postings.get(tok)
        if not bucket:
            continue
        seed_ids.update(bucket)
    if not seed_ids:
        return None
    # For each seed candidate, compute full intersection by checking
    # remaining query tokens. This is O(|seeds| * target_size) which
    # for selective queries is dramatically smaller than O(N).
    for did in seed_ids:
        inter = 0
        for tok in target_tokens:
            bucket = postings.get(tok)
            if bucket and did in bucket:
                inter += 1
        intersection_count[did] = inter
    if not intersection_count:
        return None
    target_size = len(target_tokens)
    doc_counts = slot["doc_token_count"]
    # Threshold-based pruning: a candidate can only reach Jaccard >=
    # threshold if `inter / union >= threshold`. Since
    # `union >= target_size`, this implies
    # `inter >= threshold * target_size`. Candidates whose intersection
    # is below this floor can be skipped without lookup. This is the
    # main asymptotic win — for selective queries with high threshold
    # (0.85) the candidate set drops from O(N) to O(neighbours-in-
    # vocab-clique-of-target).
    inter_floor = threshold * target_size
    # Sort descending by intersection count so we early-out as soon
    # as the head of the list dips below the floor.
    candidates = sorted(intersection_count.items(), key=lambda kv: -kv[1])
    best_doc_id: int | None = None
    best_jaccard = 0.0
    for did, inter in candidates:
        if inter < inter_floor:
            break
        b_size = doc_counts.get(did)
        if b_size is None:
            for tok in list(postings.keys()):
                postings[tok].discard(did)
            continue
        union = target_size + b_size - inter
        if union <= 0:
            continue
        score = inter / union
        if score >= threshold and score > best_jaccard:
            best_jaccard = score
            best_doc_id = did
    if best_doc_id is None:
        return None
    # Verify the row still exists and return it (stale-tolerant).
    row = get_fn(backend, best_doc_id)
    if row is None:
        # Ghost in postings but actually deleted; evict and bail. The
        # next call will rebuild a clean view eventually.
        for tok in list(postings.keys()):
            postings[tok].discard(best_doc_id)
        doc_counts.pop(best_doc_id, None)
        return None
    return row


def find_near_duplicate_source(
    backend, objective_id: int, content: str, threshold: float = 0.85
) -> dict | None:
    target = set(tokens(content))
    return _find_near_duplicate_via_index(
        backend, "source", objective_id, target, threshold,
        store.list_sources, store.get_source,
    )


def find_near_duplicate_thinking(
    backend, objective_id: int, content: str, threshold: float = 0.85
) -> dict | None:
    target = set(tokens(content))
    return _find_near_duplicate_via_index(
        backend, "thinking", objective_id, target, threshold,
        store.list_thinkings, store.get_thinking,
    )


def upsert_source(
    backend,
    objective_id: int,
    url: str | None,
    title: str | None,
    content: str,
    search_query: str | None = None,
    threshold: float = 0.85,
) -> tuple[int, bool]:
    """Insert a source with dedup. Returns (source_id, created)."""
    h = content_hash(content)
    existing = store.find_source_by_hash_in_objective(backend, objective_id, h)
    if existing is not None:
        return int(existing["id"]), False
    target_tokens = set(tokens(content))
    near = _find_near_duplicate_via_index(
        backend, "source", objective_id, target_tokens, threshold,
        store.list_sources, store.get_source,
    )
    if near:
        return int(near["id"]), False
    sid = store.insert_source(
        backend, objective_id, url, title, content, h, search_query
    )
    from . import retrieval

    retrieval.index_source(backend, sid, content)
    # Register the new doc in the inverted index AND the hash cache
    # so subsequent dedup lookups are O(1) instead of O(N).
    _register_in_index(backend, "source", objective_id, sid, target_tokens)
    store.register_hash(backend, "source", objective_id, h, sid)
    return sid, True


def upsert_thinking(
    backend,
    objective_id: int,
    content: str,
    supports_source_ids: list[int],
    author: str,
    supersedes_id: int | None = None,
    threshold: float = 0.85,
) -> tuple[int, bool]:
    h = content_hash(content)
    existing = store.find_thinking_by_hash_in_objective(backend, objective_id, h)
    if existing is not None:
        return int(existing["id"]), False
    target_tokens = set(tokens(content))
    near = _find_near_duplicate_via_index(
        backend, "thinking", objective_id, target_tokens, threshold,
        store.list_thinkings, store.get_thinking,
    )
    if near:
        return int(near["id"]), False
    tid = store.insert_thinking(
        backend, objective_id, content, h, supports_source_ids, author, supersedes_id
    )
    from . import retrieval

    retrieval.index_thinking(backend, tid, content)
    _register_in_index(backend, "thinking", objective_id, tid, target_tokens)
    store.register_hash(backend, "thinking", objective_id, h, tid)
    return tid, True
