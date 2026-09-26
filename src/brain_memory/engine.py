"""MemoryEngine——用"记忆行为"表达操作的门面。

公开接口（设计文档 §4）：encode / recall / inspect / forget / restore /
stats，外加会话级工作记忆。recall 管线为：解析 cue → 工作记忆增强 →
可选 LLM 查询扩展（cue 变体并集检索）→ 归一化多因子排序 → 可选盲评
LLM 重排 → 可选 PPR 混合 → 图扩展。所有 LLM 阶段都受配置门控并在失败
时降级到确定性路径；生命周期相关的一切（时间戳、哈希、存储、统计）
都是纯代码。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import TracebackType

import numpy as np

from brain_memory.config import MemoryConfig
from brain_memory.consolidation.conflicts import ConflictStore
from brain_memory.consolidation.consolidator import Consolidator
from brain_memory.consolidation.llm import LLMConsolidator
from brain_memory.consolidation.semantic_store import SemanticStore
from brain_memory.episodic.store import EpisodicStore
from brain_memory.embeddings.base import EmbeddingProvider
from brain_memory.embeddings.hash_embedder import HashEmbedder
from brain_memory.embeddings.openai_compatible import OpenAICompatibleEmbedder
from brain_memory.extraction.base import ExperienceParser
from brain_memory.extraction.heuristic import HeuristicExperienceParser
from brain_memory.extraction.llm import LLMExperienceParser
from brain_memory.forgetting.decay import DecaySweeper
from brain_memory.graph.links import LinkStore
from brain_memory.graph.ppr import personalized_pagerank
from brain_memory.graph.view import GraphView, parse_ref
from brain_memory.models import (
    ConflictKind,
    ConsolidationReport,
    DecayReport,
    EncodeResult,
    EngineStats,
    Episode,
    EXPLICIT_RELATIONS,
    ExtractedExperience,
    FactorScores,
    GraphSubgraph,
    MemoryConflict,
    MemoryLink,
    MemoryStatus,
    NodeKind,
    RecallResult,
    SemanticMemory,
)
from brain_memory.query.expander import LLMQueryExpander
from brain_memory.retrieval.reranker import LLMReranker
from brain_memory.retrieval.retriever import Retriever, passes_filters, semantic_view
from brain_memory.retrieval.vector_index import VectorIndex
from brain_memory.storage.db import Database
from brain_memory.working.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


def _embedding_text(content: str, entities: list[str], topics: list[str]) -> str:
    """构造嵌入输入：正文 + 结构化标签。

    拼接标签能明显改善短 cue 的召回（"minecraft 模组" 这样的 cue 能命中
    只在实体列表里提到 Minecraft 的 episode）。
    """
    tags = " ".join(dict.fromkeys(entities + topics))
    return f"{content}\n{tags}" if tags else content


class MemoryEngine:
    """持有存储、解析、嵌入、检索、巩固与工作记忆的总门面。"""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig.from_env()
        self._db = Database(self.config.db_path)
        self._store = EpisodicStore(self._db)
        self._index = VectorIndex(self._db)
        self._semantic_store = SemanticStore(self._db)
        self._semantic_index = VectorIndex(self._db, self._db.list_active_semantic_embeddings)
        self._conflict_store = ConflictStore(self._db)
        self._link_store = LinkStore(self._db)
        self._graph = GraphView(
            self._db,
            self._store,
            self._semantic_store,
            self._conflict_store,
            self._link_store,
            episodic_index=self._index,
        )
        self._embedder = self._build_embedder()
        self._parser = self._build_parser()
        self._retriever = Retriever(
            self._store,
            self._index,
            self.config,
            semantic_store=self._semantic_store,
            semantic_index=self._semantic_index,
        )
        self._expander = self._build_expander()
        self._reranker = self._build_reranker()
        self._consolidator = Consolidator(
            self._db,
            self._store,
            self._semantic_store,
            self._conflict_store,
            self.config,
            embed_fn=self._embed_text,
        )
        self._consolidator.set_llm(self._build_llm_consolidator())
        self._decay_sweeper = DecaySweeper(
            self._db, self._store, self._semantic_store, self.config
        )
        self.working = WorkingMemory()

    # -- 组件构建 ---------------------------------------------------------

    def _build_embedder(self) -> EmbeddingProvider:
        """按配置选择嵌入实现（云 API 或离线 hash）。"""
        if self.config.embedding_provider == "openai_compatible":
            return OpenAICompatibleEmbedder(
                base_url=self.config.api_base,
                api_key=self.config.api_key,
                model=self.config.embedding_model,
            )
        return HashEmbedder(dim=self.config.embedding_dim)

    def _build_parser(self) -> ExperienceParser:
        """构建经验解析器：LLM 解析器（可配）+ 启发式兜底。"""
        heuristic = HeuristicExperienceParser()
        if self.config.llm_model and self.config.llm_api_key:
            return LLMExperienceParser(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                fallback=heuristic,
            )
        return heuristic

    def _llm_ready(self) -> bool:
        """LLM 端点是否已配置（模型名 + key 均非空）。"""
        return bool(self.config.llm_model and self.config.llm_api_key)

    def _build_expander(self) -> LLMQueryExpander | None:
        """查询扩展器：auto 且配置了 LLM 才构建。"""
        if self.config.query_expansion == "auto" and self._llm_ready():
            return LLMQueryExpander(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
            )
        return None

    def _build_reranker(self) -> LLMReranker | None:
        """盲评重排器：auto 且配置了 LLM 才构建。"""
        if self.config.rerank == "auto" and self._llm_ready():
            return LLMReranker(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                timeout=self.config.rerank_timeout,
            )
        return None

    def _build_llm_consolidator(self) -> LLMConsolidator | None:
        """LLM 巩固器：配置了 LLM 才构建。"""
        if self._llm_ready():
            return LLMConsolidator(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
            )
        return None

    @property
    def embedder(self) -> EmbeddingProvider:
        """当前嵌入实现（demo/API 展示用）。"""
        return self._embedder

    # -- 核心行为 ---------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        source: str = "conversation",
        context: str | None = None,
        created_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> EncodeResult:
        """解析、嵌入并存储一段经验（append-only，见 store）。"""
        extracted = self._parser.parse(text, source=source, context=context, timestamp=created_at)
        vector = self._embed(extracted)
        episode, duplicate = self._store.add(extracted, vector, metadata=metadata)
        if not duplicate:
            self._index.invalidate()
        # 重复编码不再触发挑战（同一事件重复目击不构成新证据）
        challenge = None if duplicate else self._detect_challenge(episode)
        self.working.note_episode(episode.id)
        self.working.add_entities(episode.entities)
        return EncodeResult(episode=episode, duplicate=duplicate, challenge=challenge)

    def _detect_challenge(self, episode: Episode) -> MemoryConflict | None:
        """再巩固入口（文档 §15）：纠正信号挑战已知信念。

        打开一条冲突记录并强制该概念再巩固——检测本身不修改陈述、不降
        置信度，证据的权衡发生在再巩固时而非检测时。
        """
        if "correction" not in episode.emphasis_signals:
            return None
        # 候选目标 = 概念命中的语义记忆
        concepts = {e.casefold() for e in episode.entities} | {
            t.casefold() for t in episode.topics
        }
        targets: list[SemanticMemory] = []
        for concept in sorted(concepts):
            memory = self._semantic_store.get_by_concept(concept)
            if memory is not None and memory.status is MemoryStatus.ACTIVE:
                targets.append(memory)
        if not targets:
            # 间接纠正（"其实我改主意了"）：仅当工作记忆恰好持有一个
            # 语义命中时才采信——多个命中绝不瞎猜。
            recalled = [
                r.semantic for r in self.working.last_recall
                if r.is_semantic and r.semantic
            ]
            if len(recalled) == 1:
                targets = [recalled[0]]
        if not targets:
            return None

        conflict: MemoryConflict | None = None
        for target in targets:
            recorded = self._conflict_store.create(
                semantic_id=target.id,
                kind=ConflictKind.UNRESOLVED,
                old_version=target.version,
                statement_before=target.statement,
                trigger_episode_id=episode.id,
                trigger_kind="encode",
            )
            if conflict is None:
                conflict = recorded
        return conflict

    def recall(
        self,
        cue: str,
        *,
        k: int | None = None,
        source: str | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        require_entities: list[str] | None = None,
        touch: bool = True,
        use_working_memory: bool = True,
    ) -> list[RecallResult]:
        """为一个（可能不完整的）线索检索记忆，附解释。

        ``touch=True`` 对命中做再激活簿记：访问计数上升，喂给未来召回的
        频率因子（文档 §15）。
        """
        final_k = k or self.config.default_top_k
        parsed_cue = self._parser.parse(cue)
        boost = self._working_memory_entities() if use_working_memory else None

        variants, expanded_from, expanded_to = self._expand_query(parsed_cue, time_from, time_to)
        vectors = self._embed_variants(variants)
        limit = self.config.rerank_top_n if self._reranker is not None else final_k
        results = self._retriever.retrieve(
            variants,
            vectors,
            k=limit,
            source=source,
            time_from=expanded_from or time_from,
            time_to=expanded_to or time_to,
            require_entities=require_entities,
            entity_boost=boost,
        )
        if self._reranker is not None and len(results) >= 2:
            results = self._fuse_rerank(cue, results)
        if self.config.graph_ppr:
            results = self._blend_ppr(parsed_cue, results)
        if self.config.recall_expansion and results:
            results = self._expand_recall(
                results,
                source=source,
                time_from=expanded_from or time_from,
                time_to=expanded_to or time_to,
                require_entities=require_entities,
            )
        results = results[:final_k]
        if touch and results:
            # 再激活簿记按命中类型分流到两张表
            episodic_ids = [r.episode.id for r in results if not r.is_semantic]
            semantic_ids = [r.semantic.id for r in results if r.is_semantic and r.semantic]
            if episodic_ids:
                self._store.touch(episodic_ids)
            if semantic_ids:
                self._semantic_store.touch(semantic_ids)
        self.working.remember_recall(results)
        return results

    # -- recall 管线各阶段 -------------------------------------------------

    def _working_memory_entities(self) -> list[str] | None:
        """会话实体：偏置实体重叠因子，向当前任务相关的记忆倾斜。

        刻意只进因子、不进查询通道——陈旧的会话实体若混入 FTS 词，
        会让不相关的记忆靠新近度复活。
        """
        entities = self.working.state.active_entities[:5]
        return entities or None

    def _expand_query(
        self,
        parsed_cue: ExtractedExperience,
        time_from: datetime | None,
        time_to: datetime | None,
    ) -> tuple[list[ExtractedExperience], datetime | None, datetime | None]:
        """原始 cue + 可选的 LLM 扩展变体（仅建议性质：原始 cue 永远
        保留为独立检索通道，改写跑偏只会增加候选，不会丢候选）。"""
        variants = [parsed_cue]
        expanded_from = expanded_to = None
        if self._expander is not None:
            try:
                expansion = self._expander.expand(parsed_cue.content, self.working.snapshot())
            except Exception as exc:  # noqa: BLE001 — 扩展仅是建议
                logger.warning("query expansion failed (%s); using original cue", exc)
                expansion = None
            if expansion is not None:
                expanded_from = self._parse_iso(expansion.time_from)
                expanded_to = self._parse_iso(expansion.time_to)
                for text in [expansion.rewritten, *expansion.sub_queries[:2]]:
                    text = (text or "").strip()
                    if not text or any(text == variant.content for variant in variants):
                        continue
                    try:
                        variants.append(self._parser.parse(text))
                    except ValueError:
                        continue
        return variants, expanded_from, expanded_to

    def _embed_variants(self, variants: list[ExtractedExperience]) -> list[np.ndarray]:
        """批量嵌入全部 cue 变体（一次 API 调用）。"""
        texts = [_embedding_text(v.content, v.entities, v.topics) for v in variants]
        return list(self._embedder.embed_texts(texts))

    def _fuse_rerank(self, cue: str, results: list[RecallResult]) -> list[RecallResult]:
        """盲评 LLM 相关度融合进因子分（重排器只看内容）。

        任何失败都保持纯因子排序。
        """
        try:
            scores = self._reranker.rerank(cue, results)
        except Exception as exc:  # noqa: BLE001 — 重排仅是建议
            logger.warning("rerank failed (%s); keeping factor ranking", exc)
            return results
        mix = self.config.rerank_mix
        fused: list[RecallResult] = []
        for result, relevance in zip(results, scores):
            new_score = mix * relevance + (1.0 - mix) * result.score
            fused.append(
                result.model_copy(
                    update={
                        "llm_relevance": relevance,
                        "score": new_score,
                        "reasons": [
                            *result.reasons,
                            f"llm_relevance: {relevance:.2f} × w{mix:.2f} → {mix * relevance:.2f}",
                        ],
                    }
                )
            )
        fused.sort(key=lambda result: -result.score)
        return fused

    @staticmethod
    def _parse_iso(value: str | None) -> datetime | None:
        """解析 LLM 给出的 ISO 时间；naive 补 UTC，非法返回 None。"""
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    # -- 图感知召回（模式补全，文档 §10）--------------------------------------

    def _ppr_seeds(self, parsed_cue: ExtractedExperience) -> dict[str, float]:
        """从 cue 的实体/话题折叠出 PPR 种子概念（封顶 5 个）。"""
        concepts = list(dict.fromkeys(
            e.casefold() for e in [*parsed_cue.entities, *parsed_cue.topics]
        ))
        return {f"c:{concept}": 1.0 for concept in concepts[:5]}

    def _blend_ppr(
        self, parsed_cue: ExtractedExperience, results: list[RecallResult]
    ) -> list[RecallResult]:
        """PersonalizedPageRank 混合进因子分（研究扩展，默认关）。

        种子是 cue 的概念；PPR 质量奖励与 cue 主题结构上居中的记忆，
        与词面重叠无关。
        """
        seeds = self._ppr_seeds(parsed_cue)
        if not seeds:
            return results
        mass = personalized_pagerank(
            self._db,
            self._store,
            self._semantic_store,
            seeds=seeds,
            damping=self.config.ppr_damping,
            max_nodes=self.config.ppr_max_nodes,
        )
        if not mass:
            return results
        mix = self.config.ppr_mix

        # 既有候选：因子分与 PPR 质量加权混合
        blended: list[RecallResult] = []
        for result in results:
            kind = "s" if result.is_semantic else "e"
            ppr = mass.get(f"{kind}{result.episode.id}", 0.0)
            new_score = (1.0 - mix) * result.score + mix * ppr
            blended.append(result.model_copy(update={
                "score": new_score,
                "reasons": [*result.reasons,
                            f"graph-ppr: mass {ppr:.4f} × w{mix:.2f} → {mix * ppr:.2f}"],
            }))

        # 通道漏掉的结构性中心 episode：仅凭 PPR 质量追加（有封顶，
        # 不能淹没结果列表）
        seen = {("s" if r.is_semantic else "e", r.episode.id) for r in blended}
        added = 0
        for ref, ppr in sorted(mass.items(), key=lambda kv: -kv[1]):
            if added >= 3 or not ref.startswith("e"):
                continue
            episode_id = int(ref[1:])
            if ("e", episode_id) in seen:
                continue
            episode = self._store.get(episode_id)
            if episode is None or not episode.is_active:
                continue
            seen.add(("e", episode_id))
            added += 1
            blended.append(RecallResult(
                episode=episode,
                score=mix * ppr,
                factors=FactorScores(),
                reasons=[f"graph-ppr: mass {ppr:.4f} (structural match)"],
                expanded=True,
            ))
        blended.sort(key=lambda r: -r.score)
        return blended

    def graph_rank(self, cue: str, *, k: int = 10) -> list[tuple[str, float]]:
        """独立 PPR 排序（研究 API）：按质量降序返回节点引用与质量。

        不做混合、不带因子分。
        """
        parsed_cue = self._parser.parse(cue)
        seeds = self._ppr_seeds(parsed_cue)
        if not seeds:
            return []
        mass = personalized_pagerank(
            self._db,
            self._store,
            self._semantic_store,
            seeds=seeds,
            damping=self.config.ppr_damping,
            max_nodes=self.config.ppr_max_nodes,
        )
        return sorted(mass.items(), key=lambda kv: -kv[1])[:k]

    def _expand_recall(
        self,
        results: list[RecallResult],
        *,
        source: str | None,
        time_from: datetime | None,
        time_to: datetime | None,
        require_entities: list[str] | None,
    ) -> list[RecallResult]:
        """罚分保护下的一跳图扩展。

        扩展项得分 = 锚点 × expansion_penalty，因此原始 top-1 永远不
        会被挤掉；它们在 reasons 中携带来源并在 `expanded` 上打标。
        扩展遵守与直召回相同的过滤器——只增加上下文，绝不泄漏被
        调用方过滤掉的内容。
        """
        budget = self.config.expansion_limit
        if budget <= 0:
            return results
        penalty = self.config.expansion_penalty
        seen = {("s" if r.is_semantic else "e", r.episode.id) for r in results}

        def passes(episode: Episode) -> bool:
            return passes_filters(episode, source, time_from, time_to, require_entities)

        expanded: list[RecallResult] = []
        for anchor in results[:3]:
            if budget <= 0:
                break
            if anchor.is_semantic:
                # 语义命中：拉取其证据 episode
                memory = anchor.semantic
                for episode in self._store.get_many(memory.evidence_ids):
                    if budget <= 0 or not episode.is_active or not passes(episode):
                        continue
                    if ("e", episode.id) in seen:
                        continue
                    seen.add(("e", episode.id))
                    budget -= 1
                    expanded.append(RecallResult(
                        episode=episode,
                        score=anchor.score * penalty,
                        factors=FactorScores(entity=1.0),
                        reasons=[f"graph: evidence of S-{memory.id} ({memory.concept})"],
                        expanded=True,
                    ))
                continue

            # 情景命中：共享概念的兄弟 episode + 概念的巩固知识
            concepts = list(dict.fromkeys(
                e.casefold() for e in [*anchor.episode.entities, *anchor.episode.topics]
            ))[:3]
            for concept in concepts:
                if budget <= 0:
                    break
                ids = set(self._store.ids_for_entities([concept]))
                ids |= set(self._store.ids_for_topics([concept]))
                siblings = [
                    e for e in self._store.get_many(sorted(ids))
                    if e.is_active and ("e", e.id) not in seen and passes(e)
                ]
                for episode in siblings[:2]:
                    seen.add(("e", episode.id))
                    budget -= 1
                    expanded.append(RecallResult(
                        episode=episode,
                        score=anchor.score * penalty,
                        factors=FactorScores(entity=1.0),
                        reasons=[f"graph: shares entity '{concept}' with #{anchor.episode.id}"],
                        expanded=True,
                    ))
                memory = self._semantic_store.get_by_concept(concept)
                if (
                    memory is not None
                    and memory.status is MemoryStatus.ACTIVE
                    and ("s", memory.id) not in seen
                ):
                    seen.add(("s", memory.id))
                    budget -= 1
                    expanded.append(RecallResult(
                        episode=semantic_view(memory),
                        kind="semantic",
                        semantic=memory,
                        score=anchor.score * penalty,
                        factors=FactorScores(entity=1.0),
                        reasons=[f"graph: knowledge about '{concept}' (from #{anchor.episode.id})"],
                        expanded=True,
                    ))

        merged = [*results, *expanded]
        merged.sort(key=lambda r: -r.score)
        return merged

    # -- 图 API -------------------------------------------------------------------

    def neighborhood(self, ref: str, *, include_similar: bool = False,
                     max_per_kind: int = 6) -> GraphSubgraph:
        """'e<id>' / 's<id>' / 'c:<concept>' 的类型化一跳邻域。"""
        return self._graph.neighborhood(
            ref, include_similar=include_similar, max_per_kind=max_per_kind
        )

    def link(self, source_ref: str, relation: str, target_ref: str, *,
             weight: float = 1.0, created_by: str = "agent",
             metadata: dict | None = None) -> MemoryLink:
        """在两个实例节点之间断言一条显式关系。"""
        if relation not in EXPLICIT_RELATIONS:
            raise ValueError(
                f"relation must be one of {EXPLICIT_RELATIONS}, got {relation!r}"
            )
        source_kind, source_id, source_concept = parse_ref(source_ref)
        target_kind, target_id, target_concept = parse_ref(target_ref)
        if source_kind is NodeKind.CONCEPT or target_kind is NodeKind.CONCEPT:
            raise ValueError("explicit links connect instances, not concepts")
        # 两个端点都必须存在且活跃
        for kind, node_id in ((source_kind, source_id), (target_kind, target_id)):
            node = (
                self._store.get(node_id)
                if kind is NodeKind.EPISODE
                else self._semantic_store.get(node_id)
            )
            if node is None:
                raise ValueError(f"{kind.value} {node_id} not found")
            if not node.is_active:
                raise ValueError(f"{kind.value} {node_id} is not active")
        if (source_kind, source_id) == (target_kind, target_id):
            raise ValueError("cannot link a node to itself")
        return self._link_store.create(
            source_kind=source_kind,
            source_id=source_id,
            target_kind=target_kind,
            target_id=target_id,
            relation=relation,
            weight=weight,
            created_by=created_by,
            metadata=metadata,
        )

    def unlink(self, link_id: int) -> bool:
        """删除一条显式关系边。"""
        return self._link_store.delete(link_id)

    def related_concepts(self, concept: str, *, limit: int = 5) -> list[tuple[str, int]]:
        """活跃 episode 上与 *concept* 共现的概念。"""
        return self._graph.related_concepts(concept, limit=limit)

    # -- 内省（Inspector）--------------------------------------------------

    def strengths(self, *, limit: int = 10) -> list[dict]:
        """按计算强度升序的最弱活跃记忆（情景 + 语义）。"""
        from brain_memory.forgetting.decay import episode_strength, semantic_strength

        now = datetime.now(timezone.utc)
        half_life = self.config.recency_half_life_days
        rows: list[dict] = []
        for episode in self._store.all_active():
            rows.append({
                "kind": "episode", "id": episode.id, "ref": f"e{episode.id}",
                "strength": episode_strength(episode, now=now, half_life_days=half_life),
                "label": episode.content[:80],
            })
        for memory in self._semantic_store.list_active():
            rows.append({
                "kind": "semantic", "id": memory.id, "ref": f"s{memory.id}",
                "strength": semantic_strength(memory, now=now, half_life_days=half_life),
                "label": memory.statement[:80],
            })
        rows.sort(key=lambda row: row["strength"])
        return rows[:limit]

    def timeline(self, *, days: int = 30) -> dict:
        """按天统计编码数、新增知识与冲突数。"""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        by_day: dict[str, dict] = {}

        def absorb(rows, key: str) -> None:
            for row in rows:
                by_day.setdefault(row["day"], {})[key] = row["n"]

        absorb(self._db.count_episodes_by_day(cutoff), "episodes")
        absorb(self._db.count_semantics_by_day(cutoff), "semantics")
        absorb(self._db.count_conflicts_by_day(cutoff), "conflicts")
        return {
            "days": sorted(by_day),
            "series": [{"day": day, **by_day[day]} for day in sorted(by_day)],
        }

    def list_episodes(self, *, status: str | None = "active", limit: int = 50,
                      offset: int = 0) -> list[Episode]:
        """供检视的记忆列表（任意状态，最新在前）。"""
        if status == "active":
            rows = self._db.list_active_episodes()
        elif status == "archived":
            rows = self._db.list_archived_episodes()
        else:
            rows = self._db.list_all_episodes()
        return [self._store.row_to_episode(row) for row in rows][offset : offset + limit]

    # -- 生命周期行为 ---------------------------------------------------------------

    def forget(self, memory_id: int) -> bool:
        """软删除：先归档，物理删除是后续阶段才决策的事。"""
        archived = self._store.archive(memory_id)
        if archived:
            self._index.invalidate()
        return archived

    def restore(self, memory_id: int) -> bool:
        """从归档/遗忘状态恢复为活跃。"""
        restored = self._store.restore(memory_id)
        if restored:
            self._index.invalidate()
        return restored

    def inspect(self, memory_id: int) -> dict | None:
        """单条记忆详情 + 最相关的活跃记忆。"""
        episode = self._store.get(memory_id, with_embedding=True)
        if episode is None:
            return None
        related: list[Episode] = []
        if episode.embedding is not None:
            # 用自身向量找近邻，排除自己，最多取 3 条
            for related_id, _similarity in self._index.search(episode.embedding, 4):
                if related_id != episode.id:
                    related_episode = self._store.get(related_id)
                    if related_episode is not None:
                        related.append(related_episode)
                if len(related) >= 3:
                    break
        return {
            "episode": episode,
            "related": related,
        }

    def inspect_semantic(self, semantic_id: int) -> dict | None:
        """回答"你为什么相信这个"：陈述 + 证据 episode + 完整版本链 +
        全部冲突记录（文档 §39 原则 6）。"""
        memory = self._semantic_store.get(semantic_id)
        if memory is None:
            return None
        return {
            "semantic": memory,
            "evidence": self._store.get_many(memory.evidence_ids),
            "versions": self._semantic_store.versions(semantic_id),
            "conflicts": self._conflict_store.for_semantic(semantic_id),
        }

    def find_semantic(self, concept: str) -> SemanticMemory | None:
        """按结构键 concept 查找语义记忆。"""
        return self._semantic_store.get_by_concept(concept)

    def list_semantics(self) -> list[SemanticMemory]:
        """全部活跃语义记忆（最新更新在前）。"""
        return self._semantic_store.list_active()

    def consolidate(
        self,
        *,
        max_groups: int = 5,
        max_episodes_per_group: int = 20,
        min_support: int | None = None,
    ) -> ConsolidationReport:
        """重放 + 模式提取 → 语义记忆（文档 §11/§23）。

        基于实体/话题索引的确定性分组，组级游标增量推进，配置了 LLM 用
        LLM 提案否则用确定性统计提案。显式调用——调度交给 agent 循环或
        cron，不在引擎内起后台线程。
        """
        report = self._consolidator.consolidate(
            max_groups=max_groups,
            max_episodes_per_group=max_episodes_per_group,
            min_support=min_support,
        )
        if report.touched:
            self._semantic_index.invalidate()
        return report

    def conflicts(self, status: str | None = None) -> list[MemoryConflict]:
        """全部冲突记录，可按状态过滤（open/resolved/dismissed）。"""
        return self._conflict_store.list(status=status)

    def reconsolidate(self, semantic_id: int, *, max_episodes_per_group: int = 20) -> ConsolidationReport:
        """手动强制某个语义记忆再巩固（文档 §4: memory.reconsolidate）。"""
        memory = self._semantic_store.get(semantic_id)
        if memory is None:
            raise ValueError(f"semantic memory {semantic_id} not found")
        report = self._consolidator.consolidate(
            forced_values=[memory.concept],
            max_episodes_per_group=max_episodes_per_group,
        )
        if report.touched:
            self._semantic_index.invalidate()
        return report

    def decay(self, *, dry_run: bool = False, now: datetime | None = None) -> DecayReport:
        """遗忘扫描（文档 §14/§30）：强度驱动的归档 + 驻留驱动的遗忘。

        显式调用——调度交给 agent 循环或 cron；``dry_run=True`` 预览而不
        改动任何状态；``now`` 支持时间推演（测试/投影用）。
        """
        report = self._decay_sweeper.sweep(dry_run=dry_run, now=now)
        if not dry_run and report.transitions:
            if report.archived_episode_ids or report.forgotten_episode_ids:
                self._index.invalidate()
            if report.archived_semantic_ids or report.forgotten_semantic_ids:
                self._semantic_index.invalidate()
        return report

    # -- 统计与辅助 ------------------------------------------------------------------

    def stats(self) -> EngineStats:
        """引擎整体统计。"""
        counts = self._db.count_by_status()
        extremes = self._db.episode_extremes()
        return EngineStats(
            total_episodes=sum(counts.values()),
            active=counts.get(MemoryStatus.ACTIVE.value, 0),
            archived=counts.get(MemoryStatus.ARCHIVED.value, 0),
            forgotten=counts.get(MemoryStatus.FORGOTTEN.value, 0),
            distinct_entities=self._distinct_entity_count(),
            avg_importance=round(float(extremes["avg_importance"] or 0.0), 4),
            oldest_created_at=self._db.parse_dt(extremes["oldest"]),
            newest_created_at=self._db.parse_dt(extremes["newest"]),
            semantic_memories=self._db.semantic_count(),
            open_conflicts=self._db.count_open_conflicts(),
        )

    def _distinct_entity_count(self) -> int:
        """活跃 episode 中去重后的实体数。"""
        rows = self._db.list_active_episodes()
        seen: set[str] = set()
        for row in rows:
            for entity in self._db.loads(row["entities"], []):
                seen.add(entity.casefold())
        return len(seen)

    def _embed(self, extracted: ExtractedExperience) -> np.ndarray:
        """嵌入一条结构化经验（正文 + 标签拼接）。"""
        text = _embedding_text(extracted.content, extracted.entities, extracted.topics)
        return self._embedder.embed_texts([text])[0]

    def _embed_text(self, text: str) -> np.ndarray:
        """嵌入单条文本（巩固器写语义向量用）。"""
        return self._embedder.embed_texts([text])[0]

    # -- 生命周期 ------------------------------------------------------------------

    def close(self) -> None:
        """关闭底层存储连接。"""
        self._db.close()

    def __enter__(self) -> "MemoryEngine":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
