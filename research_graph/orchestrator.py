from __future__ import annotations

import sys
from dataclasses import dataclass, field

from . import dedup, reasoning, retrieval, storage
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
        """Canonical JSON-serializable shape used by api.research and the CLI."""
        return {
            "objective_id": int(self.objective_id),
            "mode": self.mode,
            "new_source_ids": list(self.new_source_ids),
            "new_thinking_id": self.new_thinking_id,
            "reused_thinking_id": self.reused_thinking_id,
            "supersedes_id": self.supersedes_id,
        }


class Orchestrator:
    def __init__(self, conn, search: ExternalSearch | None = None) -> None:
        self.conn = conn
        self.search = search or get_default_search()

    def research(self, objective_id: int, force_refresh: bool = False) -> ResearchResult:
        obj = storage.get_objective(self.conn, objective_id)
        if obj is None:
            raise ValueError(f"unknown objective: {objective_id}")
        existing_sources = storage.list_sources(self.conn, objective_id)
        mode = "cold_start" if not existing_sources else "memory_augmented"
        result = ResearchResult(objective_id=objective_id, mode=mode)

        do_external = mode == "cold_start" or force_refresh
        if do_external:
            hits = self.search.search(obj["question"])
            for hit in hits:
                sid, created = dedup.upsert_source(
                    self.conn,
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

        # filter -> RAG -> relevance: pick top sources for actor
        ranked = retrieval.search_sources(
            self.conn,
            obj["question"],
            objective_id=objective_id,
            top_k=5,
            min_score=0.0,
        )
        if not ranked:
            ranked = storage.list_sources(self.conn, objective_id)

        thinking_text, cited = reasoning.actor_propose(ranked, obj["question"])
        verdict = reasoning.critic_verify(self.conn, thinking_text, cited)
        if not verdict.accepted:
            result.rejected_reasons = verdict.reasons
            return result

        prior = storage.list_thinkings(self.conn, objective_id)
        supersedes = None
        if prior and result.new_source_ids:
            # new evidence arrived -> chain from latest prior thinking
            supersedes = int(prior[-1]["id"])

        tid, created = dedup.upsert_thinking(
            self.conn,
            objective_id,
            thinking_text,
            cited,
            author="actor",
            supersedes_id=supersedes,
        )
        if created:
            result.new_thinking_id = tid
            result.supersedes_id = supersedes
            storage.update_objective(self.conn, objective_id, status="researched")
        else:
            # dedup collapsed onto an existing thinking -- report which one we reused
            result.reused_thinking_id = int(tid)
            if prior:
                latest_id = int(prior[-1]["id"])
                if int(tid) != latest_id:
                    print(
                        f"[orchestrator] reused thinking {tid} is older than latest "
                        f"prior thinking {latest_id} for objective {objective_id}",
                        file=sys.stderr,
                    )
        return result
