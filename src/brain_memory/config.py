"""Engine configuration.

All knobs live here and can be set via ``MEMORY_*`` environment variables
(see ``.env.example``) or overridden programmatically.  Nothing reads the
environment outside of :meth:`MemoryConfig.from_env`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ProviderName = Literal["hash", "openai_compatible"]


class RetrievalWeights(BaseModel):
    """Weights of the multi-factor retrieval score.

    Factors are each normalized to [0, 1] *before* weighting; the final score
    is divided by the total weight so the result stays in [0, 1] even when the
    weights do not sum to 1.
    """

    semantic: float = 0.40
    keyword: float = 0.15
    recency: float = 0.15
    importance: float = 0.10
    frequency: float = 0.05
    entity: float = 0.10
    context: float = 0.05

    def total(self) -> float:
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
    db_path: str = "./memory.db"
    embedding_provider: ProviderName = "hash"
    embedding_model: str = "embedding-3"
    embedding_dim: int = 256  # hash embedder dimension; cloud providers infer from response
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""

    llm_model: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_api_key: str = ""

    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    recency_half_life_days: float = 14.0
    default_top_k: int = 5
    # How many candidates each channel (vector / keyword) contributes before ranking.
    candidate_pool_per_channel: int = 32

    # ---- Phase 2: LLM-assisted retrieval (all "auto" = active iff LLM configured) ----
    query_expansion: Literal["auto", "off"] = "auto"
    rerank: Literal["auto", "off"] = "auto"
    rerank_top_n: int = Field(default=20, ge=1)
    rerank_mix: float = Field(default=0.4, ge=0.0, le=1.0)
    rerank_timeout: float = Field(default=8.0, gt=0.0)

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = None) -> "MemoryConfig":
        """Build a config from ``MEMORY_*`` environment variables.

        Loads ``.env`` (cwd or *env_file*) first when present.  If the
        configured provider is ``openai_compatible`` but no API key is set,
        silently falls back to the hash embedder so the engine always runs.
        """
        load_dotenv(env_file)

        def get(name: str, default: str = "") -> str:
            return os.environ.get(f"MEMORY_{name}", default) or default

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
        )
