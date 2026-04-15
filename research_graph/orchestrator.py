from __future__ import annotations

import sys
from dataclasses import dataclass, field

from . import dedup, reasoning, retrieval, store
from .search import ExternalSearch, get_default_search


# Quality regression tolerance. A new Thinking whose overall quality
# is below `prior.overall - QUALITY_REGRESSION_EPSILON` does NOT
# supersede the prior live tip — it is inserted as a sibling branch
# instead (hypothesis branching). The epsilon absorbs the tiny noise
# in grounding/coverage math across re-ranked scopes so that a run
# that produces a numerically identical answer still supersedes.
QUALITY_REGRESSION_EPSILON = 0.05


@dataclass
class ResearchResult:
    objective_id: int
    mode: str  # "cold_start" or "memory_augmented"
    new_source_ids: list[int] = field(default_factory=list)
    reused_source_ids: list[int] = field(default_factory=list)
    new_thinking_id: int | None = None
    reused_thinking_id: int | None = None
    supersedes_id: int | None = None
    rejected_reasons: list[str] = field(default_factory=list)
    # iter-4 continual-research fields — always populated when a
    # Thinking is produced (new or reused). They give the daily cron
    # a measurable signal the caller can track over time.
    quality_score: dict | None = None
    actor_backend: str | None = None
    branched: bool = False

    def to_dict(self) -> dict:
        return {
            "objective_id": int(self.objective_id),
            "mode": self.mode,
            "new_source_ids": list(self.new_source_ids),
            "reused_source_ids": list(self.reused_source_ids),
            "new_thinking_id": self.new_thinking_id,
            "reused_thinking_id": self.reused_thinking_id,
            "supersedes_id": self.supersedes_id,
            "rejected_reasons": list(self.rejected_reasons),
            "quality_score": dict(self.quality_score) if self.quality_score else None,
            "actor_backend": self.actor_backend,
            "branched": bool(self.branched),
        }


class Orchestrator:
    def __init__(
        self,
        backend,
        search: ExternalSearch | None = None,
        actor_backend: reasoning.ActorBackend | None = None,
    ) -> None:
        self.backend = backend
        # keep .conn as a back-compat alias so any external caller that
        # poked at orch.conn before the rename still works against the
        # graph backend it now points at.
        self.conn = backend
        self.search = search or get_default_search()
        # iter-4: actor backend is pluggable and defaults via env var,
        # mirroring how `ExternalSearch` is resolved. Tests pass an
        # explicit fake; the daily cron picks up whatever backend is
        # registered under OPENCLAW_RESEARCH_ACTOR.
        self.actor_backend = actor_backend or reasoning.get_default_actor_backend()

    def research(
        self,
        objective_id: int,
        *,
        skip_external: bool = False,
        force_refresh: bool | None = None,
    ) -> ResearchResult:
        """Run one research pass over an objective.

        Spec contract (spec.md L57-59):
          - cold-start (no prior sources)  → external search only
          - memory-augmented (prior sources) → external search + memory blend
          - new external info supersedes / reuses old conclusions

        iter-4 structural change: supersession is now gated on a
        quality-score delta rather than on "did a new source arrive?"
        The prior behaviour was fragile — it meant day-2 always
        overwrote day-1 whenever the external fetch returned anything
        new, even if the new answer was strictly worse than the old.
        Gating on quality gives the daily cron a real feedback signal:

          - new.overall >= old.overall - QUALITY_REGRESSION_EPSILON
              → supersede (normal progress)
          - otherwise
              → insert as a SIBLING branch, do NOT supersede. The old
                live tip stays live, the new thinking is kept on record
                for later (re-evaluation, hypothesis tracking).

        This is what "더 많은 추론 파워를 넣을수록 시스템이 좋아진다"
        structurally looks like: a better actor backend produces
        thinkings with higher grounding/coverage/diversity, those
        thinkings supersede the old live tip, the live tip monotonically
        improves. A worse backend produces thinkings that are kept as
        branches but cannot regress the live answer.
        """
        obj = store.get_objective(self.backend, objective_id)
        if obj is None:
            raise ValueError(f"unknown objective: {objective_id}")
        existing_sources = store.list_sources(self.backend, objective_id)
        mode = "cold_start" if not existing_sources else "memory_augmented"
        result = ResearchResult(objective_id=objective_id, mode=mode)

        do_external = not skip_external
        if do_external:
            hits = self.search.search(obj["question"])
            for hit in hits:
                sid, created = dedup.upsert_source(
                    self.backend,
                    objective_id,
                    hit.url,
                    hit.title,
                    hit.content,
                    search_query=obj["question"],
                )
                if created:
                    result.new_source_ids.append(sid)
                else:
                    result.reused_source_ids.append(sid)

        ranked = retrieval.search_sources(
            self.backend,
            obj["question"],
            objective_id=objective_id,
            top_k=5,
            min_score=0.0,
        )
        if not ranked:
            ranked = store.list_sources(self.backend, objective_id)

        thinking_text, cited = self.actor_backend.propose(ranked, obj["question"])
        verdict = reasoning.critic_verify(self.backend, thinking_text, cited)
        if not verdict.accepted:
            result.rejected_reasons = verdict.reasons
            result.actor_backend = getattr(self.actor_backend, "name", None)
            return result

        # Compute the quality score against the ACTUAL cited source rows
        # (not just the ids) and the retrieval scope the actor saw. This
        # has to happen before insert so we can persist it atomically.
        cited_source_rows = [s for s in ranked if int(s["id"]) in set(cited)]
        quality = reasoning.score_thinking(
            thinking_text, cited_source_rows, scope_sources=ranked
        )
        result.quality_score = quality.to_dict()
        result.actor_backend = getattr(self.actor_backend, "name", None)

        prior = store.list_thinkings(self.backend, objective_id)
        # iter-4: decide supersession BEFORE the dedup/insert call,
        # using the prior live tip's persisted quality score. This way
        # the SUPERSEDES edge AND the `supersedes_id` field on the new
        # row are written atomically by `store.insert_thinking`, matching
        # pre-iter-4 provenance semantics that tests depend on.
        supersedes_candidate: int | None = None
        prior_overall: float | None = None
        if prior:
            tip = store.latest_live_thinking(self.backend, objective_id)
            if tip is not None:
                supersedes_candidate = int(tip["id"])
                po = tip.get("quality_overall")
                if po is not None:
                    prior_overall = float(po)

        allow_supersede = True
        if prior_overall is not None:
            if quality.overall < prior_overall - QUALITY_REGRESSION_EPSILON:
                allow_supersede = False

        effective_supersedes = (
            supersedes_candidate if (supersedes_candidate is not None and allow_supersede) else None
        )

        thinking_threshold = 0.98 if result.new_source_ids else 0.85
        tid, created = dedup.upsert_thinking(
            self.backend,
            objective_id,
            thinking_text,
            cited,
            author="actor",
            supersedes_id=effective_supersedes,
            threshold=thinking_threshold,
            quality_score=quality.to_dict(),
            quality_overall=float(quality.overall),
            actor_backend=getattr(self.actor_backend, "name", None),
        )

        # Branch A — a brand-new Thinking row was created. Its
        # supersedes_id field and SUPERSEDES edge were written inside
        # insert_thinking based on `effective_supersedes`.
        # Branch B/C — dedup collapsed onto an existing row. The
        # existing row already has its own stored quality_score; do
        # not overwrite.
        if created:
            result.new_thinking_id = tid
            result.supersedes_id = effective_supersedes
            result.branched = (
                supersedes_candidate is not None and effective_supersedes is None
            )
            if result.branched:
                print(
                    f"[orchestrator] quality regression on objective "
                    f"{objective_id}: new thinking {tid} overall "
                    f"{quality.overall:.3f} < prior "
                    f"{prior_overall:.3f} − {QUALITY_REGRESSION_EPSILON}; "
                    f"kept as sibling branch (no SUPERSEDES edge)",
                    file=sys.stderr,
                )
            store.update_objective(self.backend, objective_id, status="researched")
        else:
            result.reused_thinking_id = int(tid)
            tip = store.latest_live_thinking(self.backend, objective_id)
            if tip is not None:
                latest_id = int(tip["id"])
                if int(tid) != latest_id:
                    # Branch C: collapsed onto an older row; the latest
                    # live tip reused it via a REUSES edge (pre-iter-4
                    # semantics).
                    store.reuse(self.backend, latest_id, int(tid))
                    print(
                        f"[orchestrator] reused thinking {tid} is older than latest "
                        f"prior thinking {latest_id} for objective {objective_id}",
                        file=sys.stderr,
                    )
                # Branch B: tid == latest_id -> no edge, no supersedes.
            # When we reuse, the "live answer" is whatever the dedup
            # collapsed onto. Report its stored quality so the caller
            # still sees a score (not None) for every research run.
            reused_row = store.get_thinking(self.backend, int(tid))
            if reused_row is not None:
                stored = reused_row.get("quality_score")
                if stored:
                    result.quality_score = dict(stored)
                ab = reused_row.get("actor_backend")
                if ab:
                    result.actor_backend = str(ab)
        return result
