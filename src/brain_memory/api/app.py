"""可选的 FastAPI 封装：MemoryEngine + 只读 Inspector。

安装 ``pip install -e '.[server]'`` 后运行（factory 模式，导入时不构建
引擎）::

    uvicorn brain_memory.api.app:create_app --factory

HTTP 层不添加任何记忆语义——每条路由都是引擎的薄适配器。``GET /``
提供单页 Inspector（vanilla JS + 服务端 SVG，完全离线）。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from brain_memory.engine import MemoryEngine
from brain_memory.models import Episode


def _episode_dump(episode: Episode) -> dict:
    """episode 序列化（嵌入向量已在模型层排除）。"""
    return episode.model_dump(mode="json")


class EncodeRequest(BaseModel):
    """POST /encode 请求体。"""

    text: str = Field(min_length=1)
    source: str = "conversation"
    context: str | None = None


class RecallRequest(BaseModel):
    """POST /recall 请求体。"""

    cue: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=50)
    source: str | None = None
    require_entities: list[str] | None = None


def create_app(engine: MemoryEngine | None = None) -> "FastAPI":  # noqa: F821
    """构建 FastAPI 应用（全部路由为引擎的薄适配器）。"""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the API layer needs the 'server' extra: pip install -e '.[server]'"
        ) from exc

    engine = engine or MemoryEngine()
    app = FastAPI(title="Brain Memory Engine", version="0.1.0")

    # ---- 记忆行为 ----

    @app.post("/encode")
    def encode(request: EncodeRequest) -> dict:
        """编码一段经验（含挑战检测）。"""
        result = engine.encode(request.text, source=request.source, context=request.context)
        return {"episode": _episode_dump(result.episode), "duplicate": result.duplicate}

    @app.post("/recall")
    def recall(request: RecallRequest) -> list[dict]:
        """召回记忆（含因子分解与解释）。"""
        results = engine.recall(
            request.cue,
            k=request.k,
            source=request.source,
            require_entities=request.require_entities,
        )
        return [
            {
                "episode": _episode_dump(result.episode),
                "score": result.score,
                "factors": result.factors.as_dict(),
                "reasons": result.reasons,
            }
            for result in results
        ]

    @app.get("/memories/{memory_id}")
    def inspect(memory_id: int) -> dict:
        """单条情景记忆详情 + 相关记忆。"""
        inspection = engine.inspect(memory_id)
        if inspection is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return {
            "episode": _episode_dump(inspection["episode"]),
            "related": [_episode_dump(e) for e in inspection["related"]],
        }

    @app.post("/memories/{memory_id}/forget")
    def forget(memory_id: int) -> dict:
        """软删除（归档）一条记忆。"""
        if not engine.forget(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"id": memory_id, "status": "archived"}

    @app.get("/stats")
    def stats() -> dict:
        """引擎统计。"""
        return engine.stats().model_dump(mode="json")

    # ---- Inspector（只读）----

    @app.get("/")
    def inspector():
        """提供单页 Inspector 静态文件。"""
        from fastapi.responses import FileResponse

        html_path = Path(__file__).parent / "static" / "inspector.html"
        return FileResponse(html_path, media_type="text/html")

    @app.get("/api/overview")
    def overview() -> dict:
        """统计 + 配置摘要。"""
        return {
            "stats": engine.stats().model_dump(mode="json"),
            "embedding_provider": engine.embedder.name,
            "weights": engine.config.weights.model_dump(),
            "decay": {
                "archive_threshold": engine.config.decay_archive_threshold,
                "forget_after_days": engine.config.decay_forget_after_days,
            },
            "consolidation_min_support": engine.config.consolidation_min_support,
            "graph_ppr": engine.config.graph_ppr,
        }

    @app.get("/api/timeline")
    def timeline(days: int = 30) -> dict:
        """按天时间线（编码/知识/冲突计数）。"""
        return engine.timeline(days=max(1, min(days, 365)))

    @app.get("/api/memories")
    def memories(status: str = "active", limit: int = 50, offset: int = 0) -> dict:
        """双库记忆列表（带分页与状态过滤）。"""
        all_episodes = engine.list_episodes(status=None)
        window = engine.list_episodes(status=status, limit=limit, offset=offset)
        return {
            "episodes": [
                {
                    "id": e.id, "content": e.content[:160],
                    "status": e.status.value, "importance": e.importance,
                    "created_at": e.created_at.isoformat(),
                    "entities": e.entities, "access_count": e.access_count,
                }
                for e in window
            ],
            "semantics": [
                {
                    "id": m.id, "concept": m.concept, "statement": m.statement,
                    "kind": m.kind.value, "confidence": m.confidence,
                    "version": m.version, "evidence_ids": m.evidence_ids,
                }
                for m in engine.list_semantics()
            ],
            "total": len(all_episodes),
        }

    @app.get("/api/memories/episode/{memory_id}")
    def episode_detail(memory_id: int) -> dict:
        """情景记忆详情（含相关记忆）。"""
        inspection = engine.inspect(memory_id)
        if inspection is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return {
            "episode": _episode_dump(inspection["episode"]),
            "related": [_episode_dump(e) for e in inspection["related"]],
        }

    @app.get("/api/memories/semantic/{memory_id}")
    def semantic_detail(memory_id: int) -> dict:
        """语义记忆详情（陈述 + 证据 + 版本链 + 冲突）。"""
        inspection = engine.inspect_semantic(memory_id)
        if inspection is None:
            raise HTTPException(status_code=404, detail="semantic memory not found")
        return {
            "semantic": inspection["semantic"].model_dump(mode="json"),
            "evidence": [_episode_dump(e) for e in inspection["evidence"]],
            "versions": [v.model_dump(mode="json") for v in inspection["versions"]],
            "conflicts": [c.model_dump(mode="json") for c in inspection["conflicts"]],
        }

    @app.get("/api/graph/{ref}.json")
    def graph_json(ref: str) -> dict:
        """邻域子图的 JSON 形式。"""
        try:
            sub = engine.neighborhood(ref)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return sub.model_dump(mode="json")

    @app.get("/api/graph/{ref}.svg")
    def graph_svg(ref: str):
        """邻域子图的服务端 SVG 渲染。"""
        from fastapi.responses import Response

        from brain_memory.api.svg import subgraph_to_svg

        try:
            sub = engine.neighborhood(ref)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(subgraph_to_svg(sub), media_type="image/svg+xml")

    @app.get("/api/strengths")
    def strengths(limit: int = 10) -> list[dict]:
        """最弱记忆排行（遗忘视角）。"""
        return engine.strengths(limit=max(1, min(limit, 100)))

    @app.get("/api/conflicts")
    def conflicts(status: str | None = None) -> list[dict]:
        """冲突记录列表。"""
        return [c.model_dump(mode="json") for c in engine.conflicts(status=status)]

    @app.get("/api/decay/preview")
    def decay_preview() -> dict:
        """衰减 dry-run 预览。"""
        return engine.decay(dry_run=True).model_dump(mode="json")

    return app
