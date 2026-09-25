"""Forgetting: memory strength, decay sweeps, and soft lifecycle transitions.

The doc's lifecycle (§14) ends here: strong memories persist, weak ones
drift to archive, and archived memories that stay untouched eventually fall
into the FORGOTTEN terminal state — which is still soft (restore() works;
nothing is physically deleted).

Strength is computed on the fly from stored fields (no write amplification,
always consistent with the formula), reusing the retrieval layer's
normalization helpers so "what makes a memory strong" means the same thing
in ranking and in decay.
"""

from brain_memory.forgetting.decay import (
    DecaySweeper,
    episode_strength,
    last_touched,
    semantic_strength,
)

__all__ = ["DecaySweeper", "episode_strength", "semantic_strength", "last_touched"]
