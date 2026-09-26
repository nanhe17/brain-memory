"""PersonalizedPageRank：活跃记忆子图上的研究扩展排序信号。

种子是概念节点；质量沿 mentions（episode<->概念）、evidence（语义<->
episode，权重为置信度）、consolidation（概念->其语义）边扩散。按需
重算——无增量更新，这在引擎目标的规模上是正确的取舍（每轮 O(E)，
纯 Python，1 万条记忆几十毫秒）。

节点引用沿用图谱约定：'e<id>'、's<id>'、'c:<concept>'。
"""

from __future__ import annotations

from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.episodic.store import EpisodicStore
from brain_memory.storage.db import Database


def _seed_distribution(seeds: dict[str, float]) -> dict[str, float]:
    """种子权重归一化为分布。"""
    total = sum(seeds.values())
    if total <= 0:
        return {}
    return {ref: weight / total for ref, weight in seeds.items()}


def personalized_pagerank(
    db: Database,
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    *,
    seeds: dict[str, float],
    damping: float = 0.85,
    iterations: int = 30,
    tol: float = 1e-6,
    max_nodes: int = 20000,
) -> dict[str, float]:
    """PPR 质量分布（按节点引用）。确定性；总质量约等于 1。

    边双向遍历；每个节点按边权比例分配出流质量，悬挂质量（无邻居的
    节点）传送回种子分布。
    """
    seed_dist = _seed_distribution(seeds)
    if not seed_dist:
        return {}

    adjacency, node_refs = _build_graph(db, episodic_store, semantic_store, max_nodes)
    # 过滤掉被 max_nodes 裁掉的种子
    seeds = {ref: weight for ref, weight in seeds.items() if ref in adjacency}
    seed_dist = _seed_distribution(seeds)
    if not seed_dist:
        return {}

    # 幂迭代：新质量 = (1-d)*种子分布 + d*邻居分配；悬挂质量回种子
    mass = dict(seed_dist)
    for _ in range(iterations):
        nxt = {ref: (1.0 - damping) * seed_dist.get(ref, 0.0) for ref in adjacency}
        dangling = 0.0
        for ref, m in mass.items():
            neighbors = adjacency[ref]
            if not neighbors:
                dangling += m
                continue
            total_weight = sum(neighbors.values())
            for other, weight in neighbors.items():
                nxt[other] += damping * m * (weight / total_weight)
        if dangling > 0:
            for ref, share in seed_dist.items():
                nxt[ref] += damping * dangling * share
        # 收敛判定
        delta = sum(abs(nxt[ref] - mass.get(ref, 0.0)) for ref in nxt)
        mass = nxt
        if delta < tol:
            break
    return mass


def _build_graph(db: Database, episodic_store: EpisodicStore,
                 semantic_store: SemanticStore, max_nodes: int):
    """构建活跃子图的稀疏邻接（边双向）。"""
    adjacency: dict[str, dict[str, float]] = {}

    def link(a: str, b: str, weight: float) -> None:
        adjacency.setdefault(a, {})[b] = max(weight, adjacency.get(a, {}).get(b, 0.0))
        adjacency.setdefault(b, {})[a] = max(weight, adjacency.get(b, {}).get(a, 0.0))

    semantics = {m.id: m for m in semantic_store.list_active()}
    episodes = episodic_store.all_active()
    # 节点护栏：超限时裁掉 episode（保语义）
    if len(episodes) + len(semantics) > max_nodes:
        episodes = episodes[: max_nodes - len(semantics)]

    for episode in episodes:
        ref = f"e{episode.id}"
        adjacency.setdefault(ref, {})
        for value in {t.casefold() for t in [*episode.entities, *episode.topics]}:
            link(ref, f"c:{value}", 1.0)

    for memory in semantics.values():
        sref = f"s{memory.id}"
        adjacency.setdefault(sref, {})
        # 证据边：语义 -> episode（置信度加权）
        for episode_id in memory.evidence_ids:
            episode = episodic_store.get(episode_id)
            if episode is not None and episode.is_active:
                link(sref, f"e{episode_id}", max(memory.confidence, 0.05))
        # 巩固边：概念 -> 其语义
        link(f"c:{memory.concept}", sref, max(memory.confidence, 0.05))

    node_refs = set(adjacency)
    # 清理指向被裁节点的边
    for ref, neighbors in adjacency.items():
        adjacency[ref] = {n: w for n, w in neighbors.items() if n in node_refs}
    return adjacency, node_refs
