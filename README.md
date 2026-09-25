# Brain-inspired Memory Engine

Brain-inspired long-term memory architecture for AI agents — Phase 1 MVP.

A memory system for long-running agents built around a *lifecycle* (encode →
retrieve → reactivate → archive) rather than CRUD on a vector table.  It is
not a simulation of the brain; it borrows engineering-useful ideas from
hippocampus/cortex complementary learning: fast episodic capture, structured
extraction, multi-factor recall with explanations, and soft forgetting.

```
                ┌──────────────────────────────────────────┐
                │            Agent / Application           │
                └────────────────────┬─────────────────────┘
                                     │
                     MemoryEngine  (memory behaviors)
              encode │ recall │ inspect │ forget │ stats
                                     │
        ┌───────────────┬────────────┴───────────┬─────────────────┐
        ▼               ▼                        ▼                 ▼
 ┌────────────┐  ┌─────────────┐        ┌──────────────┐   ┌──────────────┐
 │  Working   │  │ Experience  │        │  Retrieval   │   │   Episodic   │
 │  Memory    │  │ Parser      │        │  vector+FTS  │   │  Store       │
 │ (session)  │  │ heur/LLM    │        │  +ranking    │   │ (append-only)│
 └────────────┘  └──────┬──────┘        └──────┬───────┘   └──────┬───────┘
                        ▼                      │                  │
                 ┌─────────────┐        ┌──────┴───────┐          │
                 │ Embedding   │◄───────┤   SQLite     │◄─────────┘
                 │ hash / API  │        │  WAL + FTS5  │
                 └─────────────┘        └──────────────┘
```

## What Phase 1 gives you

- **Episodic memory** that is append-only: experiences are immutable events,
  never merged or overwritten (pattern separation by construction).
- **Structured extraction** at encode time — entities, topics, key facts,
  emphasis signals — the data contract everything downstream consumes.
- **Hybrid, explainable retrieval**: vector + FTS5 keyword candidates ranked
  by seven normalized factors, each recall returning *why* it was recalled.
- **Working memory**: session state (goal, active entities, retrieved
  memories) that renders into a token-budgeted prompt block.
- **Soft forgetting**: archive/restore instead of deletion; decay scheduling
  arrives in a later phase.
- Cross-session persistence (SQLite), zero external services, offline by
  default.

## Quick start

```bash
pip install -e .            # hash embedder, no API key needed
# optional: pip install -e '.[server]'   for the FastAPI wrapper
```

```python
from brain_memory import MemoryEngine

engine = MemoryEngine()     # reads MEMORY_* env vars; defaults work offline

engine.encode("用户正在开发一个 Minecraft P-51 飞机模组，用 Forge 建模")
engine.encode("请记住：项目代号是 nightingale")   # importance ↑ via emphasis signals

hits = engine.recall("昨天那个飞机模组继续怎么做？", k=3)
for hit in hits:
    print(f"#{hit.episode.id} score={hit.score:.3f} :: {hit.episode.content}")
    for reason in hit.reasons:
        print("   why:", reason)

# injectable prompt block for your agent loop
print(engine.working.build_prompt_block())

# memories survive restarts (SQLite) — reopen and recall
engine.close()
```

Interactive demo (persists to `./demo_memory.db`, restart it and ask about
what you said before):

```bash
brain-memory-demo            # or: python examples/chat_memory_demo.py
```

## Configuration

All knobs are `MEMORY_*` environment variables — see [`.env.example`](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_DB_PATH` | `./memory.db` | SQLite file (`:memory:` supported) |
| `MEMORY_EMBEDDING_PROVIDER` | `hash` | `hash` (offline) or `openai_compatible` |
| `MEMORY_API_BASE` / `MEMORY_API_KEY` | Zhipu endpoint / empty | OpenAI-compatible `/embeddings` endpoint |
| `MEMORY_EMBEDDING_MODEL` | `embedding-3` | e.g. `embedding-3` (GLM) / `text-embedding-3-small` (OpenAI) |
| `MEMORY_W_*` | see `.env.example` | retrieval factor weights (auto-normalized) |
| `MEMORY_RECENCY_HALF_LIFE_DAYS` | `14` | exponential recency decay |
| `MEMORY_LLM_MODEL` / `MEMORY_LLM_API_KEY` | empty | enables the LLM experience parser, query expansion, and reranker |
| `MEMORY_QUERY_EXPANSION` | `auto` | `auto` = active iff LLM configured; `off` = force disable |
| `MEMORY_RERANK` | `auto` | same gating for the blind LLM reranker |
| `MEMORY_RERANK_TOP_N` / `MEMORY_RERANK_MIX` / `MEMORY_RERANK_TIMEOUT` | `20` / `0.4` / `8` | rerank pool size, LLM-vs-factor fusion weight, timeout |

With no API key the engine silently uses the deterministic hash embedder and
skips every LLM stage: the whole pipeline runs offline, and recall behaves
exactly like Phase 1.  Point it at a real embedding endpoint (and optionally
an LLM endpoint) for meaningful similarity and query understanding.

## Phase 2: measuring and improving retrieval

### Evaluation harness

Scenarios live in [`benchmarks/scenarios/`](benchmarks/scenarios) — each file
encodes episodes (with relative ages), fires cues with expected/forbidden
episode indexes, and tags itself (`recall` / `separation` / `temporal` /
`contamination`, mirroring the design doc's benchmarks):

```bash
brain-memory-eval --scenarios benchmarks/scenarios            # baseline table
brain-memory-eval --scenarios benchmarks/scenarios --sweep 60 # weight search
brain-memory-eval --scenarios benchmarks/scenarios --report report.md
```

The sweep samples the 7-factor weight simplex (fixed seed, reproducible),
ranks configurations by Recall@k / MRR / violations, and prints the best
config as `MEMORY_W_*` exports.  Run it with a cloud embedder configured for
meaningful tuning; the hash embedder is for CI regression.

### LLM query expansion and blind rerank

With `MEMORY_LLM_*` configured, `recall()` gains two advisory stages:

1. **Query expansion** — the cue plus the working-memory snapshot is rewritten
   into a self-contained query (resolving "那个模组" via session entities) plus
   up to two sub-queries.  Original and rewritten cues are both retrieved as
   channels, so a bad rewrite can only add candidates, never remove them.
2. **Blind rerank** — the top candidates are scored for topical relevance by
   the LLM seeing *content only* (no factor scores, no anchoring), then fused:
   `final = mix * llm_relevance + (1 - mix) * factor_score`.  Recency and
   importance keep their influence through the factor score.  Timeout or
   failure falls back to the pure factor ranking.

Session entities also bias the entity-overlap factor toward the current task
(factor-only — they never become query terms).

## The retrieval score

Each factor is normalized to [0, 1] before weighting; the score is the
weighted sum divided by total weight (stays in [0, 1]):

```
score = w1·semantic + w2·keyword(FTS bm25) + w3·recency(exp decay)
      + w4·importance + w5·frequency(log access) + w6·entity overlap + w7·context overlap
```

Every `RecallResult` carries `factors` and `reasons`, e.g.:

```
#1 score=0.494 :: 用户正在开发一个 Minecraft P-51 飞机模组…
  why: semantic: 0.57 × w0.40 → 0.23; recency: 1.00 × w0.15 → 0.15; entity: 0.40 × w0.10 → 0.04
```

## HTTP API (optional)

```bash
uvicorn brain_memory.api.app:create_app --factory
```

`POST /encode`, `POST /recall`, `GET /memories/{id}`,
`POST /memories/{id}/forget`, `GET /stats`.

## Design decisions & deviations

See [docs/DESIGN.md](docs/DESIGN.md) for the rationale, including where this
implementation deliberately deviates from the original design conversation:
append-only episodes instead of a merge arbiter, normalized factors, log
frequency, CJK-aware FTS indexing, heuristic-first importance scoring.

## Development

```bash
pip install -e '.[dev]'
pytest                 # 101 tests
```

Tests cover parsing, embeddings, storage, ranking, retrieval, working memory,
engine behaviors, the HTTP layer, the LLM expansion/rerank stages (mocked
transport), and the evaluation harness.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | episodic + working memory + hybrid retrieval + API | ✅ |
| 2 | eval harness + weight sweep, LLM query expansion, blind rerank fusion, session-entity boost | ✅ |
| 3 | replay → clustering → semantic memory (consolidation) | tables reserved |
| 4 | reconsolidation: conflict detection, versioning | planned |
| 5 | decay scheduler, memory strength, archive GC | soft delete done |
| 6-7 | memory graph, inspector UI, full benchmark suite | planned |

## License

MIT
