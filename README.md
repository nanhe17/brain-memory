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

## What Phase 2 adds

- **Evaluation harness** (`brain-memory-eval`): YAML scenarios, Recall@k /
  MRR / violation metrics, and a seeded weight-sweep over the 7-factor
  simplex. Measure first, then tune.
- **LLM query expansion**: cue + working-memory snapshot → rewritten query
  and sub-queries. Advisory only — the original cue always stays its own
  retrieval channel.
- **Blind LLM rerank**: top candidates scored on content only, fused as
  `mix * llm + (1 - mix) * factor_score`; recency/importance keep their
  voice through the factor score.
- **Session-entity boost**: working-memory entities bias the entity-overlap
  factor (factor-only — they never become query terms).

## What Phase 3 adds: consolidation

Episodes crystallize into **semantic memories** — knowledge with evidence,
confidence, and a full version trail:

```python
engine.encode("用户在学习 Java 后端，正在读 Spring 的源码")
engine.encode("用户说不想只看课程，更喜欢自学原理")
engine.encode("用户整理了 Java 的知识图谱")

report = engine.consolidate()          # replay + pattern extraction
for memory in report.created:
    print(f"S-{memory.id} ({memory.concept}): {memory.statement}")

hits = engine.recall("用户的学习方式")   # semantic hits carry kind="semantic"
inspection = engine.inspect_semantic(hits[0].semantic.id)
# -> statement + evidence episodes + version history ("why do you believe this")
```

Design properties:

- **Deterministic grouping** over the entity/topic index — no embedding
  clustering. A group needs `MEMORY_CONSOLIDATION_MIN_SUPPORT` (default 3)
  episodes before anything is proposed, and the LLM's supporting-episode
  subset must clear the same bar (over-generalization fails the gate).
- **Structural identity**: one semantic memory per concept key — never
  duplicated by textual similarity. Updates bump the version and append a
  `memory_versions` row (evidence merges, confidence smooths).
- **Honest without an LLM**: the deterministic path only states co-occurrence
  ("X appears across N memories, mainly involving …") with confidence capped
  at 0.55. Preference/fact-level statements require the LLM consolidator.
- **Incremental & budgeted**: per-concept cursors mean only groups with new
  evidence are reprocessed; `max_groups` / `max_episodes_per_group` bound
  each run. Explicit call — schedule it from your agent loop or cron.
- **Unified recall**: semantic hits join the same hybrid retrieval and show
  up in the prompt block under `[KNOWN FACTS]`, capped by
  `MEMORY_SEMANTIC_RECALL_LIMIT` so knowledge never floods episodic recall.
- **Evidence preserved**: consolidation never touches the episodes.

## What Phase 5 adds: decay & forgetting

Memory strength decides what survives; time in the archive decides what is
finally forgotten.  Everything is soft — `restore(id)` works at every stage,
nothing is ever physically deleted.

```python
report = engine.decay()              # or decay(dry_run=True) to preview
print(f"archived {report.archived_episode_ids}, "
      f"protected evidence: {report.protected_evidence_count}")
```

- **Strength** (computed on the fly, zero LLM): episodes weight importance
  0.30 / recency 0.40 / frequency 0.10 / confidence 0.20; semantics weight
  confidence 0.25 / recency 0.35 / frequency 0.10 / **evidence_support 0.30**
  (`min(1, evidence/5)`).  "请记住"-style memories carry a high static floor
  and simply never decay into the archive; knowledge with a long evidence
  chain is durable, thinly-supported knowledge eventually isn't.
- **Two mechanisms**: active → archived when strength <
  `MEMORY_DECAY_ARCHIVE_THRESHOLD` (0.25); archived → forgotten after
  `MEMORY_DECAY_FORGET_AFTER_DAYS` (90) days untouched — an archived memory
  cannot be recalled, so its recency anchor freezes and the dwell clock runs.
- **Evidence protection**: episodes cited by a *living* semantic memory are
  exempt.  Knowledge that is alive keeps its evidence alive; when the
  knowledge decays, the evidence is released to age naturally.
- **Explicit scheduling**: `engine.decay()` from your agent loop or cron;
  `dry_run=True` previews, and demo has `/decay` and `/weak`.

## What Phase 4 adds: reconsolidation

Beliefs evolve.  When new evidence challenges a consolidated statement, the
memory is re-examined and rewritten as a *temporal narrative* — never silently
overwritten, never deleted:

- **Encode-time challenge** (reconsolidation entry, doc §15): an episode with
  a correction signal ("其实…", "not … anymore") that hits a known concept —
  or the single semantic memory currently in working memory — opens a
  conflict record and forces that concept's reconsolidation.  The challenge
  edits nothing by itself; evidence is weighed at reconsolidation time.
- **Consolidation-time classification**: every update of an existing belief
  is classified (`consistent / contradiction / evolution / correction /
  context_change`).  A non-consistent verdict produces a temporal-narrative
  statement ("用户过去长期深入 Java 后端，但近期兴趣已转向 AI 方向") plus a
  resolved conflict record; `consistent` merely refines and dismisses
  unsubstantiated challenges.
- **Without an LLM**: the deterministic path tracks preference polarity
  (positive/negative/mixed from the parser's marker lexicons, recorded per
  version).  A dominant-sign shift deterministically yields an evolution
  statement ("态度信号出现变化：此前以正面为主…近期以负面为主…").
- **Gates**: rewriting an existing belief needs `MEMORY_RECONSOLIDATION_MIN_SUPPORT`
  (default 2) supporting episodes — enforced on deterministic ground in the
  orchestrator, never inside the LLM adapter.
- **Everything is inspectable**: `engine.conflicts()`,
  `engine.inspect_semantic(id)` (statement + evidence + versions + conflicts),
  `engine.reconsolidate(id)` for manual forcing, `stats().open_conflicts`,
  demo `/conflicts`.

```python
report = engine.consolidate()          # conflict-driven groups run first
for conflict in report.conflicts:
    print(f"S-{conflict.semantic_id} {conflict.kind.value}: "
          f"was: {conflict.statement_before}")
```

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
pytest                 # 163 tests
```

Tests cover parsing, embeddings, storage, ranking, retrieval, working memory,
engine behaviors, the HTTP layer, the LLM expansion/rerank stages (mocked
transport), and the evaluation harness.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | episodic + working memory + hybrid retrieval + API | ✅ |
| 2 | eval harness + weight sweep, LLM query expansion, blind rerank fusion, session-entity boost | ✅ |
| 3 | replay → semantic memory consolidation, versioning, unified recall | ✅ |
| 4 | reconsolidation: conflict detection, temporal-narrative evolution | ✅ |
| 5 | decay scheduler, memory strength, archive GC | ✅ |
| 6-7 | memory graph (conflicts are pre-shaped edges), inspector UI, long-horizon benchmark | planned |

## License

MIT
