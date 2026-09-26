"""引擎配置。

所有可调参数集中在此，可通过 ``MEMORY_*`` 环境变量（见 ``.env.example``）
或构造时覆盖设置。除 :meth:`MemoryConfig.from_env` 外任何地方都不读环境。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ProviderName = Literal["hash", "openai_compatible"]


class RetrievalWeights(BaseModel):
    """多因子检索得分的权重。

    各因子在加权*之前*先归一化到 [0, 1]；最终得分除以权重总和，因此
    即使权重不归一，得分也始终落在 [0, 1]。
    """

    semantic: float = 0.40
    keyword: float = 0.15
    recency: float = 0.15
    importance: float = 0.10
    frequency: float = 0.05
    entity: float = 0.10
    context: float = 0.05

    def total(self) -> float:
        """权重总和（作归一化分母，防零保护）。"""
        return max(
            self.semantic
            + self.keyword
            + self.recency
            + self.importance
            + self.frequency
            + self.entity
            + self.context,
            1e-9,
        )


class MemoryConfig(BaseModel):
    """引擎全局配置（存储路径 / 模型接入 / 检索与生命周期参数）。"""

    db_path: str = "./memory.db"
    embedding_provider: ProviderName = "hash"
    embedding_model: str = "embedding-3"
    embedding_dim: int = 256  # hash 嵌入的维度；云 provider 从响应推断
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""

    llm_model: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_api_key: str = ""

    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    recency_half_life_days: float = 14.0
    default_top_k: int = 5
    # 每条通道（向量 / 关键词）贡献的候选数上限
    candidate_pool_per_channel: int = 32

    # ---- Phase 2：LLM 辅助检索（"auto" = 配置了 LLM 才生效）----
    query_expansion: Literal["auto", "off"] = "auto"
    rerank: Literal["auto", "off"] = "auto"
    rerank_top_n: int = Field(default=20, ge=1)
    rerank_mix: float = Field(default=0.4, ge=0.0, le=1.0)
    rerank_timeout: float = Field(default=8.0, gt=0.0)

    # ---- Phase 3：巩固 ----
    # 概念组成型为语义记忆所需的最少 episode 数
    consolidation_min_support: int = Field(default=3, ge=1)
    # 一次召回中语义命中的上限（不能淹没情景记忆）
    semantic_recall_limit: int = Field(default=3, ge=0)
    # ---- Phase 4：再巩固 ----
    # 改写既有信念所需的支持 episode 数（低于初次巩固：再巩固是反应式的，
    # 且有旧知识作锚）
    reconsolidation_min_support: int = Field(default=2, ge=1)
    # ---- Phase 5：衰减 / 遗忘 ----
    # 记忆强度低于该值时 active -> archived
    decay_archive_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    # 归档后这么多天无任何访问 -> forgotten
    decay_forget_after_days: float = Field(default=90.0, gt=0.0)
    # ---- Phase 6：图感知召回（模式补全）----
    # 沿图边扩展召回命中（确定性、有封顶、带罚分）
    recall_expansion: bool = True
    expansion_penalty: float = Field(default=0.6, ge=0.0, le=1.0)
    expansion_limit: int = Field(default=4, ge=0)
    # ---- Phase 7：PersonalizedPageRank（研究扩展）----
    # 把 PPR 质量混入召回得分：final = (1-mix)*factor + mix*ppr
    graph_ppr: bool = False
    ppr_mix: float = Field(default=0.3, ge=0.0, le=1.0)
    ppr_damping: float = Field(default=0.85, ge=0.0, lt=1.0)
    ppr_max_nodes: int = Field(default=20000, ge=1)

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = None) -> "MemoryConfig":
        """从 ``MEMORY_*`` 环境变量构建配置。

        先加载 ``.env``（cwd 或 *env_file*）。若 provider 配置为
        ``openai_compatible`` 但没有 API key，则静默回退到 hash 嵌入器，
        保证引擎永远可运行。
        """
        load_dotenv(env_file)

        def get(name: str, default: str = "") -> str:
            return os.environ.get(f"MEMORY_{name}", default) or default

        # provider 解析：无 key 的云配置自动降级为离线 hash 嵌入
        provider = get("EMBEDDING_PROVIDER", "hash").strip().lower()
        api_key = get("API_KEY")
        if provider == "openai_compatible" and not api_key:
            provider = "hash"

        weights = RetrievalWeights(
            semantic=float(get("W_SEMANTIC", "0.40")),
            keyword=float(get("W_KEYWORD", "0.15")),
            recency=float(get("W_RECENCY", "0.15")),
            importance=float(get("W_IMPORTANCE", "0.10")),
            frequency=float(get("W_FREQUENCY", "0.05")),
            entity=float(get("W_ENTITY", "0.10")),
            context=float(get("W_CONTEXT", "0.05")),
        )

        # 文件型数据库确保父目录存在；":memory:" 跳过
        db_path = get("DB_PATH", "./memory.db")
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        return cls(
            db_path=db_path,
            embedding_provider=provider,  # type: ignore[arg-type]
            embedding_model=get("EMBEDDING_MODEL", "embedding-3"),
            embedding_dim=int(get("EMBEDDING_DIM", "256")),
            api_base=get("API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
            api_key=api_key,
            llm_model=get("LLM_MODEL", ""),
            llm_base_url=get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            llm_api_key=get("LLM_API_KEY", ""),
            weights=weights,
            recency_half_life_days=float(get("RECENCY_HALF_LIFE_DAYS", "14.0")),
            query_expansion=get("QUERY_EXPANSION", "auto").strip().lower(),
            rerank=get("RERANK", "auto").strip().lower(),
            rerank_top_n=int(get("RERANK_TOP_N", "20")),
            rerank_mix=float(get("RERANK_MIX", "0.4")),
            rerank_timeout=float(get("RERANK_TIMEOUT", "8")),
            consolidation_min_support=int(get("CONSOLIDATION_MIN_SUPPORT", "3")),
            semantic_recall_limit=int(get("SEMANTIC_RECALL_LIMIT", "3")),
            reconsolidation_min_support=int(get("RECONSOLIDATION_MIN_SUPPORT", "2")),
            decay_archive_threshold=float(get("DECAY_ARCHIVE_THRESHOLD", "0.25")),
            decay_forget_after_days=float(get("DECAY_FORGET_AFTER_DAYS", "90")),
            recall_expansion=get("RECALL_EXPANSION", "on").strip().lower() == "on",
            expansion_penalty=float(get("EXPANSION_PENALTY", "0.6")),
            expansion_limit=int(get("EXPANSION_LIMIT", "4")),
            graph_ppr=get("GRAPH_PPR", "off").strip().lower() == "on",
            ppr_mix=float(get("PPR_MIX", "0.3")),
            ppr_damping=float(get("PPR_DAMPING", "0.85")),
            ppr_max_nodes=int(get("PPR_MAX_NODES", "20000")),
        )
