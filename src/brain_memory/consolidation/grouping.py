"""Candidate group selection for replay — deterministic, index-driven.

Groups are entity/topic tags over active episodes (the ``episode_tags`` index
built at encode time).  This replaces the naive "cluster episode embeddings"
from the design conversation: conversational embeddings cluster poorly, while
tag co-occurrence is exact, explainable, and already paid for at encode time.

A group is eligible when it has at least ``min_support`` active episodes AND
at least one episode beyond the last consolidation (incremental cursor in
``consolidation_state``).  Same-value entity/topic groups collapse — entity
wins — so a value is never proposed twice per run.
"""

from __future__ import annotations

from dataclasses import dataclass

from brain_memory.storage.db import Database


@dataclass
class CandidateGroup:
    kind: str  # 'entity' | 'topic'
    value: str
    episode_count: int
    new_evidence: int
    eligible: bool = True

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.value}"


def candidate_groups(
    db: Database,
    *,
    min_support: int,
    max_groups: int,
    include_below_support: bool = False,
) -> list[CandidateGroup]:
    """Groups with new evidence, largest first.

    Groups are keyed by the *folded value* — entity:java and topic:java are
    the same concept and collapse into one candidate (entity wins as the
    representative kind, count = max across kinds).  The incremental cursor
    is likewise per value, so a consolidated concept cannot re-enter through
    its other tag kind.

    ``include_below_support`` also returns groups that exist but have fewer
    than ``min_support`` episodes — the consolidator reports those as skipped
    so a silent no-op run is explainable.
    """
    states = {
        row["value"]: int(row["episode_count"])
        for row in db.list_consolidation_states()
    }

    merged: dict[str, dict] = {}
    for row in db.tag_group_counts():
        kind, value, count = row["kind"], row["value"], int(row["n"])
        folded = value.casefold()
        entry = merged.get(folded)
        if entry is None:
            merged[folded] = {"kind": kind, "count": count}
            continue
        if entry["kind"] != "entity" and kind == "entity":
            entry["kind"] = "entity"  # entity is the representative kind
        entry["count"] = max(entry["count"], count)

    eligible: list[CandidateGroup] = []
    below: list[CandidateGroup] = []
    for folded, entry in merged.items():
        seen_before = states.get(folded, 0)
        new_evidence = max(0, entry["count"] - seen_before)
        if new_evidence < 1:
            continue
        candidate = CandidateGroup(
            kind=entry["kind"],
            value=folded,
            episode_count=entry["count"],
            new_evidence=new_evidence,
            eligible=entry["count"] >= min_support,
        )
        (eligible if candidate.eligible else below).append(candidate)

    eligible.sort(key=lambda g: (-g.episode_count, g.value))
    if not include_below_support:
        return eligible[:max_groups]
    below.sort(key=lambda g: (-g.episode_count, g.value))
    return [*eligible[:max_groups], *below[:10]]
