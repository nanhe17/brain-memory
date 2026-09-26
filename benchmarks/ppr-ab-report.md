# PPR A/B report (Phase 7 research extension)

PersonalizedPageRank blended into the factor score (`final = (1-mix)·factor +
mix·ppr`, mix = 0.3, damping = 0.85), evaluated over the 12 seed scenarios
with the deterministic hash embedder.

| configuration | Recall@5 | MRR | violations |
|---|---|---|---|
| PPR off (default) | 1.0000 | 0.8569 | 0 |
| PPR on (mix 0.3) | 1.0000 | 0.8569 | 0 |

**Reading.** On the seed scenarios (small stores, where every candidate is
already directly recalled) PPR is a no-op on metrics — as expected: PPR
rewards *structural* centrality, which only diverges from lexical matching
when the candidate pool misses structurally-related memories.  The unit
tests cover that regime directly (small candidate pools + a cue with no
lexical overlap surfacing the whole cluster).  Keep it off by default; turn
it on (`MEMORY_GRAPH_PPR=on`) when running against a real store and compare
with `brain-memory-eval` — the harness is the arbiter, not intuition.

Run date: 2026-09-25 · embedder: hash:256 · k = 5
