"""重放的候选组选择——确定性、索引驱动。

组 = 活跃 episode 的实体/话题标签（编码时建好的 ``episode_tags`` 索引）。
这替代了设计会话里朴素的"对 episode embedding 聚类"：会话嵌入的聚类
噪声很大，而标签共现精确、可解释、且编码时已经付过成本。

一个组 eligible 需要至少 ``min_support`` 条活跃 episode，且相对上次
巩固（``consolidation_state`` 增量游标）有至少一条新证据。同值的
entity/topic 组折叠为一个——同一值每次运行绝不提案两次。
"""

from __future__ import annotations

from dataclasses import dataclass

from brain_memory.storage.db import Database


@dataclass
class CandidateGroup:
    """一个候选概念组。"""

    kind: str  # 'entity' | 'topic'
    value: str
    episode_count: int
    new_evidence: int
    eligible: bool = True

    @property
    def key(self) -> str:
        """组键（kind:value，日志与跳过原因用）。"""
        return f"{self.kind}:{self.value}"


def candidate_groups(
    db: Database,
    *,
    min_support: int,
    max_groups: int,
    include_below_support: bool = False,
    forced_values: tuple[str, ...] | list[str] = (),
) -> list[CandidateGroup]:
    """有新证据的组，按规模降序。

    组按*折叠 value* 为键——entity:java 与 topic:java 是同一个概念，
    折叠为一个候选（entity 胜出为代表 kind，count 取各 kind 最大值）。
    增量游标同样按 value 记，因此已巩固的概念不可能从另一个标签 kind
    重新进入。

    ``forced_values`` 标记有 open 冲突的概念：无论如何都 eligible
    （再巩固到期，与标签计数差量无关），并排在最前。

    ``include_below_support`` 同时返回存在但少于 ``min_support`` 条
    episode 的组——巩固器会把它们报告为 skipped，让空跑可解释。
    """
    forced = {v.casefold() for v in forced_values}
    # 游标一次全部读出，按折叠 value 索引
    states = {
        row["value"]: int(row["episode_count"])
        for row in db.list_consolidation_states()
    }

    # 同 value 的 entity/topic 计数折叠：kind 取 entity 优先，count 取最大
    merged: dict[str, dict] = {}
    for row in db.tag_group_counts():
        kind, value, count = row["kind"], row["value"], int(row["n"])
        folded = value.casefold()
        entry = merged.get(folded)
        if entry is None:
            merged[folded] = {"kind": kind, "count": count}
            continue
        if entry["kind"] != "entity" and kind == "entity":
            entry["kind"] = "entity"  # entity 作为代表 kind
        entry["count"] = max(entry["count"], count)

    forced_groups: list[CandidateGroup] = []
    eligible: list[CandidateGroup] = []
    below: list[CandidateGroup] = []
    for folded, entry in merged.items():
        seen_before = states.get(folded, 0)
        new_evidence = max(0, entry["count"] - seen_before)
        is_forced = folded in forced
        if new_evidence < 1 and not is_forced:
            continue
        candidate = CandidateGroup(
            kind=entry["kind"],
            value=folded,
            episode_count=entry["count"],
            new_evidence=new_evidence,
            eligible=entry["count"] >= min_support or is_forced,
        )
        if is_forced:
            forced_groups.append(candidate)
        elif candidate.eligible:
            eligible.append(candidate)
        else:
            below.append(candidate)

    forced_groups.sort(key=lambda g: (-g.episode_count, g.value))
    eligible.sort(key=lambda g: (-g.episode_count, g.value))
    if not include_below_support:
        return (forced_groups + eligible)[:max_groups]
    below.sort(key=lambda g: (-g.episode_count, g.value))
    return [*(forced_groups + eligible)[:max_groups], *below[:10]]
