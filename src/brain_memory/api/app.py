"""Optional FastAPI wrapper around a MemoryEngine.

Install with ``pip install -e '.[server]'`` and run (factory mode, so no
engine is built at import time)::

    uvicorn brain_memory.api.app:create_app --factory

The HTTP layer adds nothing to memory semantics — every route is a thin
adapter over the engine, which remains the real API.
"""

from __future__ import annotations

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

    return app
