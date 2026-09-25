"""Interactive CLI demo: encode across turns, recall with explanations.

Run via ``brain-memory-demo`` (after ``pip install -e .``) or
``python examples/chat_memory_demo.py``.  Every line you type is encoded as
an experience; commands start with ``/``::

    /recall <cue>    retrieve memories for a cue without encoding
    /inspect <id>    show one memory and its neighbors
    /forget <id>     archive a memory (soft delete)
    /stats           memory statistics
    /prompt          show the working-memory prompt block
    /consolidate     replay + pattern extraction -> semantic memories
    /facts           list consolidated semantic memories
    /conflicts       list recorded belief conflicts
    /exit            quit

The database persists across runs — restart and ask about something you said
in a previous session to see cross-session recall.
"""

from __future__ import annotations

import argparse
import sys

from brain_memory import MemoryEngine


def _print_hit(hit, rank: int) -> None:
    episode = hit.episode
    print(f"  {rank}. #{episode.id} score={hit.score:.3f}  {episode.created_at:%Y-%m-%d %H:%M}")
    print(f"     {episode.content}")
    print(f"     entities={episode.entities} topics={episode.topics}")
    print(f"     why: {'; '.join(hit.reasons) if hit.reasons else '(no positive factors)'}")


def run(db_path: str) -> int:
    engine = MemoryEngine()
    if engine.config.db_path != db_path:
        engine.close()
        from brain_memory.config import MemoryConfig

        engine = MemoryEngine(MemoryConfig(db_path=db_path))

    print(f"Brain Memory demo — db={engine.config.db_path}, embedder={engine.embedder.name}")
    print("Type anything to remember it; /recall, /inspect, /forget, /stats, /prompt, /exit")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line == "/conflicts":
            rows = engine.conflicts()
            if not rows:
                print("  (no conflicts recorded)")
            for conflict in rows:
                target = f"S-{conflict.semantic_id}"
                print(
                    f"  #{conflict.id} {target} [{conflict.kind.value}/{conflict.status.value}] "
                    f"v{conflict.old_version}→v{conflict.resolution_version} "
                    f"via {conflict.trigger_kind}: was: {conflict.statement_before}"
                )
            continue
        if line == "/exit":
            break
        if line == "/stats":
            stats = engine.stats()
            print(
                f"  episodes: {stats.total_episodes} (active {stats.active}, "
                f"archived {stats.archived}), entities: {stats.distinct_entities}, "
                f"avg importance: {stats.avg_importance}"
            )
            continue
        if line == "/prompt":
            block = engine.working.build_prompt_block()
            print(block or "  (working memory is empty)")
            continue
        if line == "/consolidate":
            report = engine.consolidate()
            print(
                f"  groups considered: {report.groups_considered}, "
                f"created: {len(report.created)}, updated: {len(report.updated)}"
            )
            for memory in report.touched:
                print(
                    f"  S-{memory.id} [{memory.kind.value}] ({memory.concept}) "
                    f"v{memory.version} conf={memory.confidence:.2f}: {memory.statement}"
                )
            for reason in report.skipped:
                print(f"  skipped: {reason}")
            continue
        if line == "/facts":
            facts = engine.list_semantics()
            if not facts:
                print("  (no semantic memories yet — try /consolidate)")
            for memory in facts:
                print(
                    f"  S-{memory.id} [{memory.kind.value}] ({memory.concept}) "
                    f"v{memory.version} conf={memory.confidence:.2f} "
                    f"evidence={memory.evidence_ids}: {memory.statement}"
                )
            continue
        if line.startswith("/recall "):
            cue = line[len("/recall "):].strip()
            hits = engine.recall(cue, k=5, touch=False)
            if not hits:
                print("  (nothing recalled)")
            for rank, hit in enumerate(hits, 1):
                _print_hit(hit, rank)
            continue
        if line.startswith("/inspect "):
            try:
                memory_id = int(line.split()[1])
            except (IndexError, ValueError):
                print("  usage: /inspect <id>")
                continue
            inspection = engine.inspect(memory_id)
            if inspection is None:
                print("  (not found)")
                continue
            episode = inspection["episode"]
            print(
                f"  #{episode.id} [{episode.status.value}] importance={episode.importance} "
                f"confidence={episode.confidence} access={episode.access_count}"
            )
            print(f"  {episode.content}")
            print(f"  key facts: {episode.key_facts}")
            print(f"  related: {[e.id for e in inspection['related']]}")
            continue
        if line.startswith("/forget "):
            try:
                memory_id = int(line.split()[1])
            except (IndexError, ValueError):
                print("  usage: /forget <id>")
                continue
            print("  archived" if engine.forget(memory_id) else "  (not found)")
            continue

        result = engine.encode(line)
        label = "duplicate of" if result.duplicate else "encoded as"
        print(f"  {label} #{result.episode.id} (importance={result.episode.importance}, "
              f"entities={result.episode.entities})")
        if result.challenge is not None:
            print(
                f"  ⚡ challenges S-{result.challenge.semantic_id} "
                f"(was: {result.challenge.statement_before}) — run /consolidate to re-examine"
            )
        hits = engine.recall(line, k=3, touch=True)
        hits = [h for h in hits if h.episode.id != result.episode.id]
        if hits:
            print("  related memories:")
            for rank, hit in enumerate(hits, 1):
                _print_hit(hit, rank)

    engine.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain Memory Engine demo")
    parser.add_argument("--db", default="./demo_memory.db", help="sqlite db path")
    args = parser.parse_args()
    sys.exit(run(args.db))


if __name__ == "__main__":
    main()
