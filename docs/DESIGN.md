# Design Notes & Decision Record

This file records the load-bearing decisions of the Phase 1 implementation
and where it deliberately deviates from the original design conversation
(`someMemory.md`).  The original document remains the architectural north
star; this file is what the code actually does and why.

## The data contract comes first

`ExtractedExperience` (extraction/base.py → models.py) is the single boundary
through which every experience enters the system: content, entities, topics,
key_facts, emphasis_signals, importance, confidence, source, timestamp.

Everything downstream — retrieval factors, entity filtering, Phase-3
consolidation grouping — consumes these fields.  A weak parser does not break
the system (it only weakens retrieval); a changed contract would.  Treat
schema changes to it as breaking.

## Episodes are events: append-only, never merged

The original doc proposes a "pattern separation" module that decides merge vs
keep-separate.  We resolve it structurally instead:

- An episode is an immutable event.  There is no update operation.
- Encoding identical content twice does not create a second row — it bumps
  access statistics on the existing one (it is the same event re-witnessed,
  and the frequency factor benefits).
- Near-duplicate detection is a SHA-256 over whitespace-normalized content —
  exact duplicate suppression only, never fuzzy merging.
- "Similar but different" experiences therefore always stay separate rows
  (tested: 喜欢 Java / 不喜欢 Java / 重新喜欢 Java coexist).  Contamination
  at the episodic layer is impossible by construction.

Merge/split decisions belong to the semantic layer (Phase 3), where
statements are keyed structurally and carry evidence lists — not to a
fuzzy-merge arbiter over events.

## Retrieval: normalized factors, not raw quantities

Seven factors, each mapped to [0, 1] before weighting:

| factor | normalization | why |
|---|---|---|
| semantic | cosine of L2-normalized vectors | already [0,1] |
| keyword | `1/(1+bm25_rank)` | bm25 ranks are unbounded, lower=better |
| recency | `0.5 ** (age_days / half_life)` | exponential, half-life configurable |
| importance | stored [0,1] from parser heuristics | |
| frequency | `log1p(access) / log1p(cap)` | raw counts would dominate |
| entity | Jaccard(cue ∩ episode entities), casefolded | |
| context | Jaccard over topics + context tokens | |

The score is the weighted sum **divided by total weight**, so non-normalized
weight vectors still produce scores in [0, 1].  Explanations (`reasons`) are
a free by-product — the factors are computed anyway — so explainability ships
from day one instead of being retrofitted.

## FTS5 and CJK text

SQLite's unicode61 tokenizer has no CJK segmentation: a run of Han characters
becomes *one* token, so a two-character partial cue can never match.  We
store CJK characters space-separated in the FTS document and quote CJK runs
in MATCH expressions; the query parser then produces single-char token
phrases that match the indexed adjacency.  Partial cues like 「飞机模组」
resolve correctly.  Latin text is indexed and queried as normal words.

## LLMs assist; deterministic code owns the pipeline

Following Principle 5 of the design doc:

- the default parser is a zero-dependency heuristic (lexicons, quoted spans,
  emphasis-signal patterns → importance/confidence);
- the LLM parser is an optional adapter that degrades to the heuristic on
  *every* failure mode — encoding must never lose an experience;
- timestamps, hashing, storage, lifecycle state, and statistics are pure code.

Importance is heuristic-first (explicit remember-requests, corrections,
preference statements, repetition) because per-episode LLM scoring is a cost
multiplier the system does not need yet; the LLM parser refines it when
configured.

## Embeddings

Vectors are L2-normalized at write time (cosine = dot product), stored as
float32 BLOBs, and searched by brute-force numpy — millisecond latency up to
~100k episodes, zero extra infrastructure.  The dimension is asserted at
query time so a provider swap without re-encoding fails loudly instead of
silently mis-ranking.

## Forgetting is a lifecycle state, not a DELETE

Phase 1 implements active → archived (soft) → restore.  The `forgotten`
terminal state and the decay scheduler that moves memories into it are Phase
5; the enum and status column exist now so transitions stay explicit.

## Phase 2 decisions

### Measure first, then touch the retriever

The evaluation harness (`brain_memory.eval` + `benchmarks/scenarios/`) was
built before any retrieval change.  Scenarios are YAML: episodes with
relative ages, cues with expected/forbidden episode *indexes*, tags mirroring
the design doc's benchmark taxonomy (`recall` / `separation` / `temporal` /
`contamination`).  Fresh in-memory engine per run keeps ids deterministic.
Weight search samples the 7-factor simplex with a seeded Dirichlet — grid
search over 7 dimensions is hopeless, random search is reproducible.

Honest caveat baked into the docs: the seed scenarios are regression
canaries; with the hash embedder they saturate near 1.0.  Real tuning needs
a cloud embedder and harder scenarios — the harness is built for both.

### Query expansion is advisory, never authoritative

The LLM rewrites the cue using the working-memory snapshot (goal, active
entities, open questions), but the original cue always stays its own
retrieval channel.  A hallucinated or drifted rewrite can only *add*
candidates; it can never remove the ones the raw cue would have found.
Expansion may also propose a time range (ISO), which fills in — never
overrides — caller-supplied filters.  Any failure returns None; recall
degrades silently.

### Blind rerank, fused — not delegated

The reranker scores candidates seeing *content only*.  Showing it factor
scores would anchor its judgment and double-count recency/importance, which
it cannot assess from text anyway.  The final score is
`mix * llm_relevance + (1 - mix) * factor_score` (mix default 0.4): the LLM
moves memories whose *meaning* matches, while the deterministic factors keep
their voice.  Missing ids score neutral 0.5; transport/parse failures fall
back to pure factor ranking.  Both LLM stages are gated by
`MEMORY_QUERY_EXPANSION` / `MEMORY_RERANK` (default `auto` = active iff an
LLM endpoint is configured), so the no-key offline behavior is bit-identical
to Phase 1.

### Session-entity boost is factor-only

Working-memory entities bias the entity-overlap factor toward the current
task, but never become query terms.  The first implementation merged them
into the cue, which leaked them into the FTS MATCH expression — a stale
session entity resurrected fresh-but-unrelated memories on recency.  Query
terms come from cue variants alone; the boost only reweights factors.

## Phase 3 decisions

### Deterministic grouping instead of embedding clustering

The design conversation's replay step ("cluster similar episodes") is the
weakest link when taken literally: conversational episode embeddings cluster
poorly, and cluster-then-summarize reliably produces garbage knowledge.  The
implementation groups over the entity/topic index built at encode time —
exact, explainable, already paid for — and requires `min_support` (default 3)
episodes per group before anything is proposed.  Entity/topic groups with the
same folded value collapse into one candidate.

### The quality gate, not the summarizer, is the safety net

The deterministic path (no LLM configured) only ever states co-occurrence —
"X appears across N memories, mainly involving …" — with confidence hard-capped
at 0.55: it can count, it cannot judge meaning, so it never claims
preferences or facts.  The LLM path must return, alongside its statement, the
indexes of episodes that genuinely support it; a supporting subset below
`min_support` fails the gate and nothing is stored.  Over-generalization
thus degrades to "no knowledge", never to "wrong knowledge".

### Structural identity + full version trail

Semantic memories are keyed by the folded concept value — one row per
concept, never duplicated by textual similarity (pattern separation belongs
to the semantic layer, per the Phase 1 note).  Every create/update appends a
`memory_versions` row (statement, confidence, evidence, change reason), so
Phase 4's reconsolidation inherits a complete history for free.  Confidence
on update smooths (`0.5·old + 0.5·new`) and evidence unions.

### The incremental cursor is scoped to the value, not (kind, value)

The first implementation stored the consolidation cursor per
`(group_kind, group_value)`.  Because entity:java and topic:java collapse to
one candidate, the kind without a cursor always re-entered with full "new
evidence", re-consolidating the same concept and bumping versions pointlessly
(caught by the idempotency test).  The cursor is now keyed by the folded
value only.

### Unified recall via a read-only Episode view

Semantic hits carry `kind="semantic"` plus the real `SemanticMemory`, and
also an Episode-*shaped view* (content=statement, created_at=updated_at,
importance=confidence, entities=[concept]) so every existing consumer —
reranker, prompt block, HTTP layer, demo — treats hits uniformly without
branching.  The view's id lives in the semantic id space; callers must not
feed it back into episode APIs.  Semantic hits are capped
(`semantic_recall_limit`, default 3) so consolidated knowledge can never
flood episodic recall, and render under `[KNOWN FACTS]` in the prompt block.

### Evidence is never consumed

Consolidation reads episodes and links them as evidence; it archives nothing.
Provenance is what makes `inspect_semantic` (statement + evidence + versions)
an honest answer to "why do you believe this" — destroying the evidence to
save space is Phase 5 decay's job, driven by access statistics.

## Phase 4 decisions

### Two trigger layers: challenge at encode, verdict at consolidation

The neuroscience sequence — reactivation makes a memory labile, new evidence
is then weighed — maps to two distinct mechanisms.  At **encode time**, an
episode carrying a correction signal that hits a known concept (or the single
semantic memory currently in working memory) opens an *open conflict record*
and forces that concept's reconsolidation; the challenge itself edits nothing
— no statement change, no confidence penalty — because detection is not
adjudication.  At **consolidation time**, every update of an existing belief
is classified before it is applied.  Splitting detection from adjudication
keeps reactions fast without letting a single sentence rewrite knowledge.

### Temporal narratives, not overwrites

On a non-"consistent" verdict the new statement must preserve the history
("用户过去长期深入 Java 后端，但近期兴趣已转向 AI 方向") — the doc §13 shape.
The old statement is never erased: it remains in `memory_versions`, in the
conflict record's `statement_before`, and in the narrative itself.

### The deterministic path counts polarity; it does not pretend to understand

Preference direction per supporting episode (positive/negative/mixed, from
the parser's marker lexicons; negative markers are scrubbed before the
positive check so 「不喜欢」 is not miscounted as 「喜欢」) is recorded in each
version's metadata.  A dominant-sign shift across versions — over the three
signs pos/neg/mixed, with a first observation never counting as a shift —
deterministically produces an evolution statement with the before/after
tally embedded.  Subtler conflicts wait for the LLM path; the deterministic
path states only what it counted.

### The support gate lives in the orchestrator, not the LLM adapter

Reconsolidation uses a lower threshold than initial consolidation (2 vs 3,
configurable): it is reactive, anchored on an existing belief, and often
triggered by an explicit user correction.  Crucially, the gate is enforced
in `Consolidator._repropose` on deterministic ground — an LLM adapter (or
any duck-typed stand-in) cannot bypass it.  This was not free: the first
implementation trusted the adapter, and a test with a misbehaving mock
walked straight through.

### Conflicts are records now, graph edges later

`memory_conflicts` stores both endpoints (semantic memory + triggering
episode), the pre-state (`old_version`, `statement_before`), and the
resolution — exactly the shape a `contradicts`/`updates` edge needs.  Open
conflicts force their concept's group due on every consolidate run; resolved
and dismissed records stay as history.  Open conflicts never degrade recall
ranking — they are visible through `inspect_semantic` / `conflicts()` /
`stats().open_conflicts`, and punishment, if any, is the reconsolidation's
job.

## Phase 5 decisions

### Two mechanisms instead of two thresholds

The doc's strength formula alone cannot drive the full lifecycle: importance
and confidence form a *static floor* (0.30·importance + 0.20·confidence for
typical chit-chat ≈ 0.20), so any "forget" threshold below that floor never
fires, and any above it makes state jumps noisy.  Hence the split:
**active → archived is strength-driven** (the floor means explicitly
important memories never decay into the archive — the doc's "strong memories
persist", for free); **archived → forgotten is dwell-driven** (an archived
memory cannot be recalled, its recency anchor freezes, and after
`decay_forget_after_days` it reaches the terminal state).  Each mechanism is
individually explainable, and transitions are strictly sequential: a memory
archived in one sweep can only be forgotten by a later one.

### Strength is computed, never stored

No strength column, no write amplification, no staleness — the formula reads
only already-stored fields and reuses the retrieval layer's normalization
helpers, so "strong" means the same thing in ranking and in decay.  The
weights are a calibration unit in code (they move together), not user
configuration; the two thresholds are configurable because they encode
policy (how forgetful the deployment wants to be).

### Evidence protection as emergent dynamics

Episodes cited by a living semantic memory are exempt from the sweep.  The
doc's CSL story then falls out by itself: episodic details age unless they
keep being used, knowledge with a long evidence chain is durable, thin
knowledge eventually decays — and when knowledge decays, its evidence is
*released* to age, instead of being deleted with it.  Emotional salience is
deliberately absent as a factor: emphasis signals are already folded into
importance at encode time ("请记住" → +0.35), which is exactly the
persistence behavior the doc wanted from salience.

### FORGOTTEN is a state, not a deletion

Data stays; `restore()` works from every state; FTS rows remain (status
filters guard retrieval and consolidation grouping).  Physical deletion is
out of scope for this project's v1 lifecycle — if storage pressure ever
demands it, it becomes an explicit, auditable GC job behind the same
interfaces, not a decay side effect.

## Phase 6 decisions

### The graph already existed; the missing piece was the read model

Auditing the stored data before building anything: `episode_tags` is an
episode↔concept bipartite graph, `evidence_ids` are `derived_from` edges,
`memory_conflicts` were designed (Phase 4) as pre-shaped `contradicts` edges.
Materializing all of that into one big edge table would have created three
sync points and a consistency risk for zero information gain.  Hence:
**derive on demand, materialize only what cannot be derived** — and the only
such things are edges an agent or user explicitly asserts (`memory_links`,
closed relation vocabulary, instance endpoints only).

### Concept nodes are the connective tissue

Without entity/concept pseudo-nodes the graph is a pile of disconnected
instances; with them, "Java —related— Spring" and "which episodes mention
Java" are the same traversal.  Concept nodes fold entity and topic kinds by
value (they were always the same concept — the Phase 3 cursor bug taught
that), and co-occurrence between concepts is a weighted self-join over the
tags index rather than a stored edge.

### Expansion is bounded, penalized, and filter-obeying

Graph-aware recall appends one-hop neighbors (siblings via shared entity,
evidence of recalled knowledge, the concept's consolidated belief) at
`anchor × expansion_penalty`.  Three invariants: the original top-1 can
never be displaced (penalty < 1); expanded entries carry provenance reasons
and the `expanded` flag (a context hit must say it is one); and expansion
applies the caller's recall filters (source/time/require_entities) — it may
add context, never leak what was filtered out.

One honest calibration note: with brute-force candidate generation over a
small store, every memory weakly matches every cue, so expansion only
*changes results* when the candidate pool is smaller than the store (the
regime that mimics large deployments).  The expansion tests therefore run
with `candidate_pool_per_channel=1-2`; the seed scenario verifies no
regression at default settings.

### Lifecycle is a read-time concern

Edges whose endpoints left the active state are filtered during traversal —
no cascade deletes, no link-GC.  `restore()` brings relationships back by
itself, consistent with every other soft transition in this system.  Global
graph algorithms (PersonalizedPageRank à la HippoRAG) remain a Phase 7
research extension; one-hop expansion with provenance is what recall can
explain today.

## Known limitations (accepted for Phase 1)

- Heuristic entity extraction misses bare lowercase latin tokens (e.g.
  `nightingale`) unless quoted, capitalized, digit-bearing, or in the
  lexicon.  The LLM parser closes this gap when configured.
- Single-writer SQLite; concurrent processes are out of scope until
  consolidation (Phase 3) runs as a background job — then WAL + retry policy
  gets revisited.
- The vector index is an in-memory matrix rebuilt on write invalidation;
  incremental updates are unnecessary at this scale.
- No decay scheduling: memories do not weaken over time yet, they only move
  between explicit states.

## Concurrency model

One connection, one RLock, WAL journal.  Reads and writes are serialized
through the lock (correctness first; throughput is not a Phase 1 problem at
human conversation rates).
