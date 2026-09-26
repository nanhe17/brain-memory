"""混合检索：向量 + 关键词双通道，并集、因子打分、排序。

设计：两条通道是*候选生成器*（面向召回）；归一化多因子打分器随后对
并集排序（面向精度）。这与设计文档的"Retrieval Engine"同构，并让每条
通道保持简单：向量通道覆盖转述改写，关键词通道覆盖精确词与嵌入会
模糊掉的稀有实体。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from brain_memory.config import MemoryConfig
from brain_memory.episodic.store import EpisodicStore
from brain_memory.models import Episode, ExtractedExperience, RecallResult
from brain_memory.retrieval import ranking
from brain_memory.retrieval.vector_index import VectorIndex


def build_fts_match_expr(cue: ExtractedExperience) -> str:
    """构造"引号短语 OR"形式的 MATCH 表达式。

    索引侧把中文字符空格分隔存储（unicode61 分词器没有 CJK 分词能力），
    查询侧对每段中文连读加引号——查询解析器随后产出单字符词元短语，
    与索引中的相邻单字符词元匹配。
    """
    terms: list[str] = []
    for token in ranking._LATIN_RE.findall(cue.content.lower()):
        if len(token) >= 2:
            terms.append(token)
    for entity in cue.entities:
        for token in ranking._LATIN_RE.findall(entity.lower()):
            if len(token) >= 2:
                terms.append(token)
    quoted = [_quote(term) for term in terms]
    for run in ranking._CJK_RE.findall(cue.content):
        if len(run) >= 2:
            quoted.append(_quote(run))
    # 去重并保持顺序
    seen: set[str] = set()
    unique = [q for q in quoted if not (q in seen or seen.add(q))]
    return " OR ".join(unique)


def _quote(term: str) -> str:
    """FTS5 引号包裹，内部引号双写转义。"""
    return '"' + term.replace('"', '""') + '"'


def passes_filters(episode, source: str | None, time_from: datetime | None,
                   time_to: datetime | None, require_entities: list[str] | None) -> bool:
    """召回共用过滤器——图扩展的候选同样适用，因此扩展永远不会泄漏
    被调用方过滤掉的记忆。"""
    if not episode.is_active:
        return False
    if source is not None and episode.source != source:
        return False
    if time_from is not None and episode.created_at < time_from:
        return False
    if time_to is not None and episode.created_at > time_to:
        return False
    if require_entities:
        have = {e.casefold() for e in episode.entities}
        wanted = {e.casefold() for e in require_entities}
        if not wanted & have:
            return False
    return True


def semantic_view(memory):
    """语义记忆的 Episode 形只读视图（见 RecallResult）。"""
    return Episode(
        id=memory.id,
        content=memory.statement,
        content_hash=f"semantic-{memory.id}",
        entities=[memory.concept],
        topics=[],
        created_at=memory.updated_at,
        importance=memory.confidence,
        confidence=memory.confidence,
        access_count=memory.access_count,
        last_accessed=memory.last_accessed,
        status=memory.status,
        metadata={"kind": "semantic", "concept": memory.concept},
    )


class Retriever:
    """混合检索器：情景通道 + 语义通道。"""

    def __init__(
        self,
        store: EpisodicStore,
        index: VectorIndex,
        config: MemoryConfig,
        semantic_store=None,
        semantic_index: VectorIndex | None = None,
    ) -> None:
        self._store = store
        self._index = index
        self._config = config
        self._semantic_store = semantic_store
        self._semantic_index = semantic_index

    def retrieve(
        self,
        cue_variants: list[ExtractedExperience],
        vectors: list[np.ndarray],
        k: int | None = None,
        *,
        source: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        require_entities: list[str] | None = None,
        entity_boost: list[str] | None = None,
    ) -> list[RecallResult]:
        """为一个或多个 cue 变体（如原文 + LLM 扩展）排序候选。

        通道在各变体间取并集：语义相似取最大值，关键词取最好的（最小
        的）bm25 rank。

        ``entity_boost`` 只喂给实体重叠*因子*——绝不会变成查询词。会话
        实体让排序偏向当前任务相关的记忆，同时不拓宽候选通道（查询词
        只来自变体本身）。
        """
        k = k or self._config.default_top_k
        pool = self._config.candidate_pool_per_channel
        now = datetime.now(timezone.utc)

        # 双通道并集收集候选：id -> (最好语义相似度, 最好 bm25 rank)
        candidates: dict[int, tuple[float, float]] = {}
        for variant, vector in zip(cue_variants, vectors):
            for episode_id, similarity in self._index.search(vector, pool):
                best_sim, best_rank = candidates.get(episode_id, (-1.0, math.inf))
                candidates[episode_id] = (max(best_sim, similarity), best_rank)
            match_expr = build_fts_match_expr(variant)
            if match_expr:
                for episode, rank in self._store.fts_search(match_expr, pool):
                    best_sim, best_rank = candidates.get(episode.id, (-1.0, math.inf))
                    candidates[episode.id] = (best_sim, min(best_rank, rank))
        if not candidates:
            return []

        # 因子集合：跨变体取并集（扩展解析出指代 -> P-51 时能纠正方向）
        cue_entities: set[str] = set()
        cue_context: set[str] = set()
        for variant in cue_variants:
            cue_entities |= {e.casefold() for e in variant.entities}
            cue_context |= {t.casefold() for t in variant.topics}
            if variant.context:
                cue_context |= ranking._context_tokens(variant.context)
        if entity_boost:
            cue_entities |= {e.casefold() for e in entity_boost}
        cue_entities = set(sorted(cue_entities)[:12])

        episodic = self._rank_episodic(
            candidates, cue_entities, cue_context, now,
            source=source, time_from=time_from, time_to=time_to,
            require_entities=require_entities,
        )
        # 未接语义通道时直接返回情景结果
        if self._semantic_store is None or self._semantic_index is None:
            episodic.sort(key=lambda result: -result.score)
            return episodic[:k]

        semantic = self._rank_semantic(
            cue_variants, vectors, cue_entities, cue_context, now, pool
        )
        merged = self._merge_with_cap(episodic, semantic)
        return merged[:k]

    # -- 情景通道 ---------------------------------------------------------

    def _rank_episodic(
        self,
        candidates: dict[int, tuple[float, float]],
        cue_entities: set[str],
        cue_context: set[str],
        now: datetime,
        *,
        source: str | None,
        time_from: datetime | None,
        time_to: datetime | None,
        require_entities: list[str] | None,
    ) -> list[RecallResult]:
        """情景候选打分（不做截断，交给上层合并）。"""
        episodes = self._store.get_many(list(candidates.keys()))
        episodes = [
            episode
            for episode in episodes
            if passes_filters(episode, source, time_from, time_to, require_entities)
        ]
        results: list[RecallResult] = []
        for episode in episodes:
            semantic_sim, bm25_rank = candidates[episode.id]
            keyword = ranking.normalize_bm25(bm25_rank) if bm25_rank < math.inf else 0.0
            factors = ranking.compute_factors(
                episode=episode,
                semantic=semantic_sim,
                keyword=keyword,
                cue_entities=cue_entities,
                cue_context=cue_context,
                now=now,
                half_life_days=self._config.recency_half_life_days,
            )
            score = ranking.score(factors, self._config.weights)
            results.append(
                RecallResult(
                    episode=episode,
                    kind="episodic",
                    score=score,
                    factors=factors,
                    reasons=ranking.explain(factors, self._config.weights),
                )
            )
        return results

    # -- 语义通道 ------------------------------------------------------------

    def _rank_semantic(
        self,
        cue_variants: list[ExtractedExperience],
        vectors: list[np.ndarray],
        cue_entities: set[str],
        cue_context: set[str],
        now: datetime,
        pool: int,
    ) -> list[RecallResult]:
        """语义记忆候选：语义向量 + 语义 FTS，同样双通道取并集。"""
        candidates: dict[int, tuple[float, float]] = {}
        for variant, vector in zip(cue_variants, vectors):
            for semantic_id, similarity in self._semantic_index.search(vector, pool):
                best_sim, best_rank = candidates.get(semantic_id, (-1.0, math.inf))
                candidates[semantic_id] = (max(best_sim, similarity), best_rank)
            match_expr = build_fts_match_expr(variant)
            if match_expr:
                for memory, rank in self._semantic_store.fts_search(match_expr, pool):
                    best_sim, best_rank = candidates.get(memory.id, (-1.0, math.inf))
                    candidates[memory.id] = (best_sim, min(best_rank, rank))
        if not candidates:
            return []

        memories = self._semantic_store.get_many(list(candidates.keys()))
        results: list[RecallResult] = []
        for memory in memories:
            if not memory.status.value == "active":
                continue
            view = semantic_view(memory)
            semantic_sim, bm25_rank = candidates[memory.id]
            keyword = ranking.normalize_bm25(bm25_rank) if bm25_rank < math.inf else 0.0
            factors = ranking.compute_factors(
                episode=view,
                semantic=semantic_sim,
                keyword=keyword,
                cue_entities=cue_entities,
                cue_context=cue_context,
                now=now,
                half_life_days=self._config.recency_half_life_days,
            )
            score = ranking.score(factors, self._config.weights)
            results.append(
                RecallResult(
                    episode=view,
                    kind="semantic",
                    semantic=memory,
                    score=score,
                    factors=factors,
                    reasons=ranking.explain(factors, self._config.weights),
                )
            )
        return results

    def _merge_with_cap(
        self, episodic: list[RecallResult], semantic: list[RecallResult]
    ) -> list[RecallResult]:
        """两通道按得分交错合并，但语义记忆绝不能淹没召回列表
        （``semantic_recall_limit``）。"""
        merged = [*episodic, *semantic]
        merged.sort(key=lambda result: -result.score)
        kept: list[RecallResult] = []
        semantic_kept = 0
        for result in merged:
            if result.is_semantic:
                if semantic_kept >= self._config.semantic_recall_limit:
                    continue
                semantic_kept += 1
            kept.append(result)
        return kept
