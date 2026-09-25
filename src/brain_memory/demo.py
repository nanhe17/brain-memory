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
    /decay           run the forgetting sweep (archive weak, forget stale)
    /weak            dry-run: preview what the next decay would archive
    /graph <ref>     typed one-hop neighborhood (e1 / s1 / c:java), mermaid
    /link <a> <rel> <b>
                     assert an explicit relationship between two nodes
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
        if line == "/decay" or line == "/weak":
            report = engine.decay(dry_run=(line == "/weak"))
            label = "would transition" if report.dry_run else "transitions"
            print(
                f"  swept {report.swept_episodes} episodes / "
                f"{report.swept_semantics} semantics, "
                f"{label}: archive {len(report.archived_episode_ids)}+"
                f"{len(report.archived_semantic_ids)}, "
                f"forget {len(report.forgotten_episode_ids)}+"
                f"{len(report.forgotten_semantic_ids)}, "
                f"protected evidence: {report.protected_evidence_count}"
            )
            continue
        if line.startswith("/graph "):
            ref = line[len("/graph "):].strip()
            try:
                sub = engine.neighborhood(ref)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            from brain_memory.graph.render import render_mermaid

            print(render_mermaid(sub))
            continue
        if line.startswith("/link "):
            parts = line.split()
            if len(parts) != 4:
                print("  usage: /link <src> <relation> <dst>  (e.g. /link e1 related_to e2)")
                continue
            try:
                link = engine.link(parts[1], parts[2], parts[3])
            except ValueError as exc:
                print(f"  {exc}")
                continue
            print(f"  linked #{link.id}: {parts[1]} -[{link.relation}]-> {parts[3]}")
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
