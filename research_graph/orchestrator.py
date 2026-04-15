from __future__ import annotations

import sys
from dataclasses import dataclass, field

from . import dedup, reasoning, retrieval, store
from .search import ExternalSearch, get_default_search


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
        }


class Orchestrator:
    def __init__(self, backend, search: ExternalSearch | None = None) -> None:
        self.backend = backend
        # keep .conn as a back-compat alias so any external caller that
        # poked at orch.conn before the rename still works against the
        # graph backend it now points at.
        self.conn = backend
        self.search = search or get_default_search()

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

        The previous implementation tied "do we fetch external?" to a
        `force_refresh` flag whose default `False` meant memory-augmented
        runs **never** fetched external sources, breaking AC6/AC7/AC9.
        That made the scheduled "매일 9시 자동 리서치" use case a no-op
        after day 1 because no new evidence ever arrived.

        New contract: external fetch is the default for every research
        run. The `mode` label is purely descriptive (cold-start vs
        memory-augmented based on existing sources). Callers can opt out
        of the external fetch with `skip_external=True` (e.g. for offline
        replay or when the search backend is known stale).

        `force_refresh` is kept as a deprecated alias so existing
        callers don't break — when set explicitly to True it still forces
        external (which is now the default anyway), and when set to False
        it still forces external (rather than the broken old behaviour of
        suppressing external on memory-augmented runs).
        """
        obj = store.get_objective(self.backend, objective_id)
        if obj is None:
            raise ValueError(f"unknown objective: {objective_id}")
        existing_sources = store.list_sources(self.backend, objective_id)
        mode = "cold_start" if not existing_sources else "memory_augmented"
        result = ResearchResult(objective_id=objective_id, mode=mode)

        # External fetch policy (spec L57-59):
        #   - default: ALWAYS fetch external; new evidence is what the
        #     daily cron exists to bring in.
        #   - opt out via skip_external=True for offline / replay paths.
        #   - force_refresh kept as alias; it never suppresses external.
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

        thinking_text, cited = reasoning.actor_propose(ranked, obj["question"])
        verdict = reasoning.critic_verify(self.backend, thinking_text, cited)
        if not verdict.accepted:
            result.rejected_reasons = verdict.reasons
            return result

        prior = store.list_thinkings(self.backend, objective_id)
        # supersedes_candidate is the prior latest LIVE thinking we WOULD
        # point a new thinking at if one gets created this run. It is
        # only used in the created=True branch below; the created=False
        # branches MUST ignore it (see mutual-exclusion comment at the
        # tail).
        #
        # We use store.latest_live_thinking() rather than `prior[-1]`
        # because list-order-as-time-proxy is fragile across backends
        # (see store.latest_live_thinking docstring). The supersession-
        # tip helper follows :SUPERSEDES edges and breaks ties by
        # created_at, which is correct regardless of how the backend
        # hands out node ids.
        supersedes_candidate: int | None = None
        if prior and result.new_source_ids:
            tip = store.latest_live_thinking(self.backend, objective_id)
            if tip is not None:
                supersedes_candidate = int(tip["id"])

        thinking_threshold = 0.98 if result.new_source_ids else 0.85
        tid, created = dedup.upsert_thinking(
            self.backend,
            objective_id,
            thinking_text,
            cited,
            author="actor",
            supersedes_id=supersedes_candidate,
            threshold=thinking_threshold,
        )

        # The three tail branches are mutually exclusive and must match
        # ResearchResult.to_dict() exactly so provenance on disk matches
        # what we report back to the caller.
        #
        #   Branch A (created=True):
        #     A brand-new Thinking row was inserted. store.insert_thinking
        #     already wrote (new_tid)-[:SUPERSEDES]->(supersedes_candidate)
        #     inside upsert_thinking, so we only need to echo
        #     supersedes_id into the result. No REUSES edge -- the new row
        #     IS the live conclusion, nothing "collapsed."
        #
        #   Branch B (created=False, tid == latest prior thinking):
        #     Dedup collapsed onto the row that is already the latest live
        #     thinking in this objective. This is a pure no-op run: the
        #     live answer has not moved. Record `reused_thinking_id` for
        #     the caller, write NO edges. A REUSES edge here would be a
        #     self-loop (forbidden by store.reuse) or spurious provenance.
        #
        #   Branch C (created=False, tid != latest prior thinking):
        #     Dedup collapsed onto an OLDER row. The latest live thinking
        #     in this objective traced back to `tid` because new evidence
        #     pointed that way. Write (latest_id)-[:REUSES]->(tid) with
        #     the convention "latest -> reused_ancestor". Do NOT set
        #     supersedes_id: no new live thinking was produced this run.
        if created:
            # Branch A
            result.new_thinking_id = tid
            result.supersedes_id = supersedes_candidate
            store.update_objective(self.backend, objective_id, status="researched")
        else:
            result.reused_thinking_id = int(tid)
            tip = store.latest_live_thinking(self.backend, objective_id)
            if tip is not None:
                latest_id = int(tip["id"])
                if int(tid) != latest_id:
                    # Branch C: collapse onto older row.
                    store.reuse(self.backend, latest_id, int(tid))
                    print(
                        f"[orchestrator] reused thinking {tid} is older than latest "
                        f"prior thinking {latest_id} for objective {objective_id}",
                        file=sys.stderr,
                    )
                # Branch B: tid == latest_id -> no edge, no supersedes.
        return result
