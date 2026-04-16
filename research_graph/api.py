"""Top-level Python API used by the OpenClaw plugin + CLI.

The plugin talks to the graph through this module so everything flows
through one canonical entrypoint — the same entrypoint the Python tests
hit — guaranteeing plugin behaviour and library behaviour stay in sync.

Three entrypoints:

  * `research(db, objective_id)` — run one research cycle against an
    existing objective. Thin wrapper over Orchestrator; returns a
    `ResearchResult.to_dict()` payload.

  * `research_journey(db, topic, question, ...)` — the **continual-
    research** entrypoint. Resolves (or creates) the topic+objective,
    snapshots the live-memory state BEFORE the run, runs the cycle,
    snapshots the state AFTER the run, computes a delta, and returns a
    rich payload that a plugin caller (OpenClaw) can turn directly into
    a user-facing narrative without having to hand-walk the graph.

  * `thinking_history(db, objective_id)` — walk the SUPERSEDES chain
    from the current live tip down, plus any sibling (branched) live
    thinkings. This is "show the user how the answer evolved" without
    the caller having to know about internal edge labels.
"""
from __future__ import annotations

from pathlib import Path

from . import store
from .graph import get_default_backend
from .orchestrator import Orchestrator


def research(db_path: str | Path, objective_id: int, force_refresh: bool = False) -> dict:
    backend = get_default_backend(db_path)
    try:
        orch = Orchestrator(backend)
        result = orch.research(objective_id, force_refresh=force_refresh)
        return result.to_dict()
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Continual-research user journey
# ---------------------------------------------------------------------------


_SNIPPET_MAX = 240


def _snippet(text: str | None, limit: int = _SNIPPET_MAX) -> str:
    if not text:
        return ""
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def _memory_snapshot(backend, objective_id: int) -> dict:
    """Single snapshot of the live-memory state of one objective.

    Captures everything the plugin narrator needs: current live tip
    (id + quality + snippet + actor backend), total counts of sources
    and thinkings, and the set of sibling live thinkings (= live nodes
    that are NOT the quality-best live tip).

    Why sibling enumeration matters: iter-4's hypothesis-branching
    semantics let a weaker run insert a thinking as a sibling without
    unseating the live tip. For the plugin user journey we want to
    show the operator "here are N other live answers we kept around"
    so they can inspect or reconcile them later.
    """
    obj = store.get_objective(backend, int(objective_id))
    if obj is None:
        return {
            "objective_id": int(objective_id),
            "exists": False,
            "live_thinking_id": None,
            "live_quality_overall": None,
            "live_content_snippet": "",
            "live_actor_backend": None,
            "source_count": 0,
            "thinking_count": 0,
            "live_count": 0,
            "sibling_live_thinkings": [],
        }

    sources = store.list_sources(backend, int(objective_id))
    thinkings = store.list_thinkings(backend, int(objective_id))

    # Which thinkings are still live (no one has superseded them)?
    superseded_ids: set[int] = set()
    for t in thinkings:
        sup = backend.out_neighbors(
            int(t["id"]), rel=store.REL_SUPERSEDES, label=store.LABEL_THINKING
        )
        for tgt in sup:
            superseded_ids.add(int(tgt["id"]))
    live_rows = [t for t in thinkings if int(t["id"]) not in superseded_ids]

    tip = store.latest_live_thinking(backend, int(objective_id))
    tip_id = int(tip["id"]) if tip else None

    siblings = []
    for t in live_rows:
        if tip_id is not None and int(t["id"]) == tip_id:
            continue
        siblings.append(
            {
                "thinking_id": int(t["id"]),
                "quality_overall": t.get("quality_overall"),
                "actor_backend": t.get("actor_backend"),
                "created_at": t.get("created_at"),
                "content_snippet": _snippet(t.get("content")),
            }
        )
    siblings.sort(
        key=lambda s: (
            -(float(s["quality_overall"] or -1.0)),
            str(s.get("created_at") or ""),
            int(s["thinking_id"]),
        )
    )

    return {
        "objective_id": int(objective_id),
        "exists": True,
        "question": obj.get("question"),
        "status": obj.get("status"),
        "live_thinking_id": tip_id,
        "live_quality_overall": (tip or {}).get("quality_overall"),
        "live_content_snippet": _snippet((tip or {}).get("content")),
        "live_actor_backend": (tip or {}).get("actor_backend"),
        "source_count": len(sources),
        "thinking_count": len(thinkings),
        "live_count": len(live_rows),
        "sibling_live_thinkings": siblings,
    }


def _resolve_or_create_topic_objective(
    backend, topic_name: str, question: str
) -> tuple[int, bool, int, bool]:
    """Return (topic_id, topic_existed, objective_id, objective_existed).

    Exact-match semantics on topic name and objective question text.
    Never merges close-but-not-equal names/questions — spec L75-76
    requires that ambiguity be handled conservatively by the system,
    not by auto-merging.
    """
    topic = store.get_topic_by_name(backend, topic_name)
    if topic is None:
        topic_id = store.create_topic(backend, topic_name)
        topic_existed = False
    else:
        topic_id = int(topic["id"])
        topic_existed = True

    objective_id: int | None = None
    for row in store.query_objectives(backend, topic_id=topic_id):
        if row.get("question") == question:
            objective_id = int(row["id"])
            break
    if objective_id is None:
        objective_id = store.create_objective(backend, topic_id, question)
        objective_existed = False
    else:
        objective_existed = True
    return int(topic_id), topic_existed, int(objective_id), objective_existed


def _classify_outcome(
    run: dict,
    before: dict,
    after: dict,
) -> tuple[str, float | None, str]:
    """Derive (outcome, quality_delta, narrative) from raw result + snapshots.

    Outcome taxonomy (mutually exclusive):
      "cold_start"   - first run for this objective (no prior thinking).
      "improved"     - new thinking superseded prior, live tip moved,
                       quality delta >= 0.
      "branched"     - new thinking kept as sibling, live tip unchanged
                       because quality would have regressed.
      "reused"       - dedup collapsed onto an existing thinking (no new
                       row created).
      "rejected"     - critic rejected the actor proposal.
      "unchanged"    - shouldn't normally happen, defensive fallback.
    """
    delta: float | None = None
    before_q = before.get("live_quality_overall")
    after_q = after.get("live_quality_overall")
    if before_q is not None and after_q is not None:
        delta = round(float(after_q) - float(before_q), 4)

    if run.get("rejected_reasons"):
        return (
            "rejected",
            delta,
            "Critic rejected the actor proposal: "
            + "; ".join(run["rejected_reasons"]),
        )
    if run.get("mode") == "cold_start":
        return (
            "cold_start",
            delta,
            "First research run for this objective; "
            f"new live tip quality_overall={after_q}.",
        )
    if run.get("new_thinking_id") is not None and run.get("branched"):
        return (
            "branched",
            delta,
            "New thinking kept as sibling branch (quality regression); "
            f"live tip unchanged at {before.get('live_thinking_id')}.",
        )
    if run.get("new_thinking_id") is not None and run.get("supersedes_id") is not None:
        return (
            "improved",
            delta,
            f"New thinking {run['new_thinking_id']} superseded prior tip "
            f"{run['supersedes_id']}; quality delta={delta}.",
        )
    if run.get("reused_thinking_id") is not None:
        return (
            "reused",
            delta,
            "Dedup collapsed onto existing thinking "
            f"{run['reused_thinking_id']}; no new answer this run.",
        )
    return ("unchanged", delta, "Research run produced no graph delta.")


def research_journey(
    db_path: str | Path,
    topic: str,
    question: str,
    *,
    skip_external: bool = False,
    force_refresh: bool = False,
) -> dict:
    """Continual-research user-journey entrypoint for the plugin.

    Flow:
      1. Resolve or create (topic, objective) via exact-match lookup.
      2. Snapshot memory BEFORE the run — live tip id/quality/snippet,
         sibling thinkings, source/thinking counts.
      3. Run one orchestrator cycle. The orchestrator is the iter-4
         quality-gated supersession engine, so a weak run becomes a
         sibling branch automatically.
      4. Snapshot memory AFTER the run.
      5. Classify the outcome (cold_start | improved | branched |
         reused | rejected | unchanged) and compute quality_delta.
      6. Return a rich payload the plugin can turn directly into
         narrative without re-walking the graph.
    """
    backend = get_default_backend(db_path)
    try:
        # Resolve-or-create happens inside the `before` transaction so
        # the JSON backend gets one consistent write boundary. BUT the
        # snapshot we report as `memory_before` must reflect the state
        # the caller perceives as "before my research call" — if the
        # objective did NOT pre-exist, `memory_before.exists` is False
        # regardless of the fact that we just created the row. We get
        # that by taking the snapshot FIRST (when possible) and only
        # falling back to post-create snapshot shape when the objective
        # is brand new.
        existing_topic = store.get_topic_by_name(backend, topic)
        pre_existing_obj_id: int | None = None
        if existing_topic is not None:
            for row in store.query_objectives(
                backend, topic_id=int(existing_topic["id"])
            ):
                if row.get("question") == question:
                    pre_existing_obj_id = int(row["id"])
                    break
        if pre_existing_obj_id is not None:
            before = _memory_snapshot(backend, pre_existing_obj_id)
        else:
            # Objective doesn't exist yet — report the canonical
            # "empty memory" snapshot without an id.
            before = {
                "objective_id": None,
                "exists": False,
                "live_thinking_id": None,
                "live_quality_overall": None,
                "live_content_snippet": "",
                "live_actor_backend": None,
                "source_count": 0,
                "thinking_count": 0,
                "live_count": 0,
                "sibling_live_thinkings": [],
            }

        topic_id, topic_existed, objective_id, objective_existed = (
            _resolve_or_create_topic_objective(backend, topic, question)
        )
    finally:
        backend.close()

    # Re-open a fresh backend for the run so the JSON backend's write
    # boundary is clean and any in-process indexes start from the
    # on-disk truth (matches the pattern `research()` already uses).
    backend = get_default_backend(db_path)
    try:
        orch = Orchestrator(backend)
        run_result = orch.research(
            objective_id, skip_external=skip_external, force_refresh=force_refresh
        ).to_dict()
    finally:
        backend.close()

    backend = get_default_backend(db_path)
    try:
        after = _memory_snapshot(backend, objective_id)
    finally:
        backend.close()

    outcome, delta, narrative = _classify_outcome(run_result, before, after)

    return {
        "topic": {
            "id": int(topic_id),
            "name": topic,
            "existed_before": bool(topic_existed),
        },
        "objective": {
            "id": int(objective_id),
            "question": question,
            "existed_before": bool(objective_existed),
        },
        "memory_before": before,
        "run": run_result,
        "memory_after": after,
        "delta": {
            "quality_overall_delta": delta,
            "outcome": outcome,
            "narrative": narrative,
        },
    }


# ---------------------------------------------------------------------------
# Supersession-chain history
# ---------------------------------------------------------------------------


def thinking_history(db_path: str | Path, objective_id: int) -> dict:
    """Walk the supersession chain from the live tip down + enumerate
    sibling live thinkings (hypothesis branches).

    The caller gets a structured "how did this answer evolve" view:

      {
        "objective_id": int,
        "live_tip_id":  int | None,
        "chain": [
          {"thinking_id", "quality_overall", "actor_backend",
           "created_at", "content_snippet", "supersedes_id",
           "citation_count"},
          ...                                     # tip first, oldest last
        ],
        "siblings": [ ...same shape... ],         # live, not on the chain
        "warnings": [str, ...],                   # cycle / fanout notices
      }

    Traversal cap = 64 so a corrupted graph with a cycle cannot stall
    the plugin; cycles are reported via `warnings`.
    """
    backend = get_default_backend(db_path)
    try:
        obj = store.get_objective(backend, int(objective_id))
        if obj is None:
            return {
                "objective_id": int(objective_id),
                "exists": False,
                "live_tip_id": None,
                "chain": [],
                "siblings": [],
                "warnings": [],
            }

        tip = store.latest_live_thinking(backend, int(objective_id))
        if tip is None:
            return {
                "objective_id": int(objective_id),
                "exists": True,
                "live_tip_id": None,
                "chain": [],
                "siblings": [],
                "warnings": [],
            }

        chain: list[dict] = []
        seen: set[int] = set()
        warnings: list[str] = []
        cursor: dict | None = tip
        hop = 0
        while cursor is not None and hop < 64:
            tid = int(cursor["id"])
            if tid in seen:
                warnings.append(
                    f"supersession cycle detected at thinking {tid}; truncated"
                )
                break
            seen.add(tid)
            chain.append(_format_history_entry(backend, cursor))
            predecessors = backend.out_neighbors(
                tid, rel=store.REL_SUPERSEDES, label=store.LABEL_THINKING
            )
            if not predecessors:
                cursor = None
            else:
                # Follow the first edge — supersession is 1:1 by construction
                # (orchestrator only ever emits one SUPERSEDES per created
                # row), but if a hand-edited graph has multi-fanout we log
                # a warning and follow the first edge deterministically.
                if len(predecessors) > 1:
                    warnings.append(
                        f"thinking {tid} has {len(predecessors)} SUPERSEDES edges; following first"
                    )
                cursor = store.get_thinking(backend, int(predecessors[0]["id"]))
            hop += 1

        # Sibling live thinkings: not on the chain, not superseded by
        # anything. These are the hypothesis branches.
        on_chain_ids = {entry["thinking_id"] for entry in chain}
        all_thinkings = store.list_thinkings(backend, int(objective_id))
        superseded_ids: set[int] = set()
        for t in all_thinkings:
            sup = backend.out_neighbors(
                int(t["id"]), rel=store.REL_SUPERSEDES, label=store.LABEL_THINKING
            )
            for tgt in sup:
                superseded_ids.add(int(tgt["id"]))
        siblings = []
        for t in all_thinkings:
            tid = int(t["id"])
            if tid in on_chain_ids:
                continue
            if tid in superseded_ids:
                continue
            siblings.append(_format_history_entry(backend, t))
        siblings.sort(
            key=lambda e: (
                -(float(e.get("quality_overall") or -1.0)),
                str(e.get("created_at") or ""),
                int(e["thinking_id"]),
            )
        )

        return {
            "objective_id": int(objective_id),
            "exists": True,
            "live_tip_id": int(tip["id"]),
            "chain": chain,
            "siblings": siblings,
            "warnings": warnings,
        }
    finally:
        backend.close()


def _format_history_entry(backend, thinking_row: dict) -> dict:
    tid = int(thinking_row["id"])
    cites = backend.out_neighbors(tid, rel=store.REL_CITES, label=store.LABEL_SOURCE)
    return {
        "thinking_id": tid,
        "quality_overall": thinking_row.get("quality_overall"),
        "actor_backend": thinking_row.get("actor_backend"),
        "created_at": thinking_row.get("created_at"),
        "content_snippet": _snippet(thinking_row.get("content")),
        "supersedes_id": thinking_row.get("supersedes_id"),
        "citation_count": len(cites),
    }
