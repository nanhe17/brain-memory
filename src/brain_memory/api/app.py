"""Optional FastAPI wrapper around a MemoryEngine + read-only Inspector.

Install with ``pip install -e '.[server]'`` and run (factory mode, so no
engine is built at import time)::

    uvicorn brain_memory.api.app:create_app --factory

The HTTP layer adds nothing to memory semantics — every route is a thin
adapter over the engine.  ``GET /`` serves the single-page Inspector
(vanilla JS + server-side SVG, fully offline).

The HTTP layer adds nothing to memory semantics — every route is a thin
adapter over the engine, which remains the real API.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from brain_memory.engine import MemoryEngine
from brain_memory.models import Episode


def _episode_dump(episode: Episode) -> dict:
    return episode.model_dump(mode="json")


class EncodeRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "conversation"
    context: str | None = None


class RecallRequest(BaseModel):
    cue: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=50)
    source: str | None = None
    require_entities: list[str] | None = None


def create_app(engine: MemoryEngine | None = None) -> "FastAPI":  # noqa: F821
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the API layer needs the 'server' extra: pip install -e '.[server]'"
        ) from exc

    engine = engine or MemoryEngine()
    app = FastAPI(title="Brain Memory Engine", version="0.1.0")

    @app.post("/encode")
    def encode(request: EncodeRequest) -> dict:
        result = engine.encode(request.text, source=request.source, context=request.context)
        return {"episode": _episode_dump(result.episode), "duplicate": result.duplicate}

    @app.post("/recall")
    def recall(request: RecallRequest) -> list[dict]:
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
        inspection = engine.inspect(memory_id)
        if inspection is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return {
            "episode": _episode_dump(inspection["episode"]),
            "related": [_episode_dump(episode) for episode in inspection["related"]],
        }

    @app.post("/memories/{memory_id}/forget")
    def forget(memory_id: int) -> dict:
        if not engine.forget(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"id": memory_id, "status": "archived"}

    @app.get("/stats")
    def stats() -> dict:
        return engine.stats().model_dump(mode="json")

    # ---- Inspector (read-only) ----

    @app.get("/")
    def inspector():
        from fastapi.responses import FileResponse

        html_path = Path(__file__).parent / "static" / "inspector.html"
        return FileResponse(html_path, media_type="text/html")

    @app.get("/api/overview")
    def overview() -> dict:
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
        return engine.timeline(days=max(1, min(days, 365)))

    @app.get("/api/memories")
    def memories(status: str = "active", limit: int = 50, offset: int = 0) -> dict:
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
        inspection = engine.inspect(memory_id)
        if inspection is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return {
            "episode": _episode_dump(inspection["episode"]),
            "related": [_episode_dump(e) for e in inspection["related"]],
        }

    @app.get("/api/memories/semantic/{memory_id}")
    def semantic_detail(memory_id: int) -> dict:
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
        try:
            sub = engine.neighborhood(ref)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return sub.model_dump(mode="json")

    @app.get("/api/graph/{ref}.svg")
    def graph_svg(ref: str):
        from fastapi.responses import Response

        from brain_memory.api.svg import subgraph_to_svg

        try:
            sub = engine.neighborhood(ref)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(subgraph_to_svg(sub), media_type="image/svg+xml")

    @app.get("/api/strengths")
    def strengths(limit: int = 10) -> list[dict]:
        return engine.strengths(limit=max(1, min(limit, 100)))

    @app.get("/api/conflicts")
    def conflicts(status: str | None = None) -> list[dict]:
        return [c.model_dump(mode="json") for c in engine.conflicts(status=status)]

    @app.get("/api/decay/preview")
    def decay_preview() -> dict:
        return engine.decay(dry_run=True).model_dump(mode="json")

    return app
