"""
songkeeper.py — SonglineMemory Dream Cycle
JSG Labs / Geldrich Corp

Phase 2.

The SongKeeper is the background consolidation process that runs during
idle time. It mimics biological memory consolidation — the way sleep
strengthens important neural pathways and lets unused ones fade.

What it does each cycle:
    1. DECAY    — weakens Songline edges that haven't been walked recently
    2. PRUNE    — deletes Songline edges that have decayed to the floor
    3. SCORE    — recalculates relevancy scores on all active Landmarks
    4. PROMOTE  — graduates Landmarks that pass all three gates into Lore
    5. RETIRE   — soft-removes promoted Landmarks from the active graph
    6. DIARY    — writes a human-readable DREAMS.md entry for the cycle

Promotion gates (all three must pass):
    relevancy_score    >= PROMOTE_RELEVANCY_MIN  (default 0.6)
    recall_count       >= PROMOTE_RECALL_MIN     (default 3)
    unique_query_count >= PROMOTE_UNIQUE_MIN     (default 2)

Decay model:
    Every Songline edge that was NOT walked since the last cycle loses
    DECAY_RATE strength. Edges at or below PRUNE_FLOOR are deleted.
    Edges that WERE walked are left alone — walk_songline() already
    strengthened them on traversal.

Relevancy scoring:
    Relevancy is recalculated each cycle from three signals:
        - recall_count       (how often recalled — normalized)
        - unique_query_count (how many distinct contexts — normalized)
        - recency            (how recently last accessed — decays with age)
    Weighted combination → new relevancy_score, clamped 0.0–1.0.

Dream Diary:
    After every cycle, a human-readable entry is appended to DREAMS.md.
    Shows exactly what was strengthened, decayed, pruned, promoted.
    Trust through transparency.

Batch import source tagging:
    Landmarks placed via bulk import (e.g. Gemini wiki ingestion) carry
    a source tag in the concept_label or via the tags field in lore_entries.
    SongKeeper treats them identically — same gates, same decay, same diary.
    No special handling needed. The pipeline handles source attribution.

Scheduling:
    Run via cron on 851Office-1. Recommended: nightly at 2am.
    Add to crontab with:
        crontab -e
    Then add this line:
        0 2 * * * python3 /mnt/SonglineMemory/songkeeper.py >> /mnt/SonglineMemory/songkeeper.log 2>&1

Zero external dependencies. Pure Python stdlib + math.
"""

import sqlite3
import math
import os
from datetime import datetime, timedelta

from core_memory import (
    DB_NAME,
    LORE_DB_NAME,
    PROMOTE_RELEVANCY_MIN,
    PROMOTE_RECALL_MIN,
    PROMOTE_UNIQUE_MIN,
    get_all_active_landmarks,
    get_all_songlines,
    update_landmark_relevancy,
    retire_landmark,
    promote_to_lore,
    decay_songline,
    delete_songline,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DECAY_RATE       = 0.05   # strength lost per cycle on unwalked edges
PRUNE_FLOOR      = 0.10   # edges at or below this get deleted
DREAMS_PATH      = "/mnt/SonglineMemory/DREAMS.md"
CYCLE_LOG_PATH   = "/mnt/SonglineMemory/songkeeper.log"

# Relevancy scoring weights — must sum to 1.0
W_RECALL         = 0.40   # weight for recall frequency
W_UNIQUE         = 0.35   # weight for unique query diversity
W_RECENCY        = 0.25   # weight for how recently accessed

# Normalization caps
RECALL_CAP       = 20     # 20+ recalls = full recall score
UNIQUE_CAP       = 10     # 10+ unique queries = full unique score
RECENCY_DAYS_CAP = 30     # older than 30 days = zero recency score


# ---------------------------------------------------------------------------
# RECENCY SCORING
# ---------------------------------------------------------------------------

def _recency_score(last_accessed: str | None) -> float:
    """
    Returns 0.0–1.0 recency score based on last_accessed timestamp.
    Full score if accessed today. Zero if never accessed or older than cap.
    Linear decay between.
    """
    if not last_accessed:
        return 0.0
    try:
        last = datetime.strptime(last_accessed, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0.0

    age_days = (datetime.now() - last).days
    if age_days >= RECENCY_DAYS_CAP:
        return 0.0
    return 1.0 - (age_days / RECENCY_DAYS_CAP)


# ---------------------------------------------------------------------------
# RELEVANCY RECALCULATION
# ---------------------------------------------------------------------------

def _calculate_relevancy(landmark: dict) -> float:
    """
    Recalculates relevancy score from three signals.
    Weighted combination, clamped 0.0–1.0.
    """
    recall_score = min(landmark['recall_count'] / RECALL_CAP, 1.0)
    unique_score = min(landmark['unique_query_count'] / UNIQUE_CAP, 1.0)
    recency      = _recency_score(landmark.get('last_accessed'))

    score = (
        W_RECALL  * recall_score +
        W_UNIQUE  * unique_score +
        W_RECENCY * recency
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# PROMOTION GATE CHECK
# ---------------------------------------------------------------------------

def _passes_promotion_gates(landmark: dict) -> bool:
    """All three gates must pass — not a majority, all three."""
    return (
        landmark['relevancy_score']    >= PROMOTE_RELEVANCY_MIN and
        landmark['recall_count']       >= PROMOTE_RECALL_MIN    and
        landmark['unique_query_count'] >= PROMOTE_UNIQUE_MIN
    )


# ---------------------------------------------------------------------------
# RECENTLY WALKED EDGE IDS
# ---------------------------------------------------------------------------

def _get_recently_walked_ids(since: datetime) -> set:
    """
    Returns set of Songline IDs walked since the given datetime.
    Used to skip decay on edges that were recently traversed.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        SELECT id FROM songlines
        WHERE last_walked IS NOT NULL AND last_walked >= ?
    """, (since_str,))
    ids = {row['id'] for row in cursor.fetchall()}
    conn.close()
    return ids


# ---------------------------------------------------------------------------
# DREAM DIARY WRITER
# ---------------------------------------------------------------------------

def _write_dream_diary(cycle_log: dict) -> None:
    """
    Appends a human-readable entry to DREAMS.md after each cycle.
    This is the transparency log — shows exactly what SongKeeper did.
    """
    now = cycle_log['timestamp']
    lines = []

    lines.append(f"\n## [{now}] — SongKeeper Cycle\n")

    # Stats summary
    lines.append(f"**Landmarks active:** {cycle_log['landmarks_active']}")
    lines.append(f"**Songlines total:** {cycle_log['songlines_total']}")
    lines.append("")

    # Decay & prune
    lines.append(f"### Edge Maintenance")
    lines.append(f"- Edges decayed: {cycle_log['edges_decayed']}")
    lines.append(f"- Edges pruned (strength ≤ {PRUNE_FLOOR}): {cycle_log['edges_pruned']}")
    lines.append("")

    # Relevancy updates
    lines.append(f"### Relevancy Updates")
    if cycle_log['relevancy_updates']:
        for item in cycle_log['relevancy_updates']:
            direction = "↑" if item['new'] > item['old'] else "↓"
            lines.append(
                f"- {direction} **{item['label']}** "
                f"{item['old']:.3f} → {item['new']:.3f}"
            )
    else:
        lines.append("- No significant changes.")
    lines.append("")

    # Promotions to Lore
    lines.append(f"### Promoted to Lore")
    if cycle_log['promoted']:
        for item in cycle_log['promoted']:
            lines.append(
                f"- ✦ **{item['label']}** "
                f"(recalls: {item['recalls']} | "
                f"unique queries: {item['unique']} | "
                f"relevancy: {item['relevancy']:.3f})"
            )
    else:
        lines.append("- Nothing promoted this cycle.")
    lines.append("")

    lines.append("---")

    entry = "\n".join(lines)

    # Create DREAMS.md if it doesn't exist, append if it does
    mode = "a" if os.path.exists(DREAMS_PATH) else "w"
    with open(DREAMS_PATH, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# DREAMS.md — SongKeeper Dream Diary\n")
            f.write("*A transparent record of every consolidation cycle.*\n")
            f.write("*What was strengthened. What faded. What was remembered forever.*\n")
        f.write(entry)

    print(f"DREAM DIARY: Entry written to {DREAMS_PATH}")


# ---------------------------------------------------------------------------
# MAIN CYCLE
# ---------------------------------------------------------------------------

def run_cycle(verbose: bool = True) -> dict:
    """
    Runs one full SongKeeper consolidation cycle.

    Returns a summary dict of what was done — useful for testing
    and for the Dream Diary writer.

    Steps:
        1. Load active state
        2. Decay unwalked Songline edges
        3. Prune dead edges
        4. Recalculate relevancy scores
        5. Promote landmarks that pass all three gates
        6. Write Dream Diary entry
    """
    now       = datetime.now()
    now_str   = now.strftime("%Y-%m-%d %H:%M:%S")
    cycle_log = {
        'timestamp':        now_str,
        'landmarks_active': 0,
        'songlines_total':  0,
        'edges_decayed':    0,
        'edges_pruned':     0,
        'relevancy_updates': [],
        'promoted':         [],
    }

    if verbose:
        print(f"\n[SongKeeper] Cycle starting — {now_str}")
        print(f"[SongKeeper] DBs: {DB_NAME} | {LORE_DB_NAME}\n")

    # ------------------------------------------------------------------
    # STEP 1 — Load active state
    # ------------------------------------------------------------------
    landmarks = get_all_active_landmarks()
    songlines = get_all_songlines()

    cycle_log['landmarks_active'] = len(landmarks)
    cycle_log['songlines_total']  = len(songlines)

    if verbose:
        print(f"[SongKeeper] {len(landmarks)} active landmarks | "
              f"{len(songlines)} songlines")

    # ------------------------------------------------------------------
    # STEP 2 — Decay unwalked Songline edges
    # Edges walked within the last 24 hours are skipped.
    # ------------------------------------------------------------------
    since_yesterday   = now - timedelta(hours=24)
    recently_walked   = _get_recently_walked_ids(since_yesterday)
    edges_to_decay    = [s for s in songlines if s['id'] not in recently_walked]

    if verbose:
        print(f"[SongKeeper] Edges to decay: {len(edges_to_decay)} "
              f"({len(recently_walked)} walked recently, skipped)")

    decayed_count = 0
    pruned_ids    = []

    for edge in edges_to_decay:
        new_strength = decay_songline(edge['id'], DECAY_RATE)
        decayed_count += 1
        if new_strength <= PRUNE_FLOOR:
            pruned_ids.append(edge['id'])

    cycle_log['edges_decayed'] = decayed_count

    # ------------------------------------------------------------------
    # STEP 3 — Prune dead edges
    # ------------------------------------------------------------------
    for sid in pruned_ids:
        delete_songline(sid)
        if verbose:
            print(f"[SongKeeper] PRUNED edge {sid} (strength ≤ {PRUNE_FLOOR})")

    cycle_log['edges_pruned'] = len(pruned_ids)

    # ------------------------------------------------------------------
    # STEP 4 — Recalculate relevancy scores
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[SongKeeper] Recalculating relevancy scores...")

    for lm in landmarks:
        old_score = lm['relevancy_score']
        new_score = _calculate_relevancy(lm)

        if abs(new_score - old_score) >= 0.01:  # only update if meaningful change
            update_landmark_relevancy(lm['id'], new_score)
            cycle_log['relevancy_updates'].append({
                'label': lm['concept_label'],
                'old':   old_score,
                'new':   new_score,
            })
            lm['relevancy_score'] = new_score  # update in-memory for gate check below

            if verbose:
                direction = "↑" if new_score > old_score else "↓"
                print(f"[SongKeeper] {direction} '{lm['concept_label']}' "
                      f"{old_score:.3f} → {new_score:.3f}")

    # ------------------------------------------------------------------
    # STEP 5 — Promote landmarks that pass all three gates
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[SongKeeper] Checking promotion gates...")

    for lm in landmarks:
        if _passes_promotion_gates(lm):
            lore_id = promote_to_lore(lm)
            retire_landmark(lm['id'])

            cycle_log['promoted'].append({
                'label':     lm['concept_label'],
                'recalls':   lm['recall_count'],
                'unique':    lm['unique_query_count'],
                'relevancy': lm['relevancy_score'],
                'lore_id':   lore_id,
            })

            if verbose:
                print(f"[SongKeeper] ✦ PROMOTED: '{lm['concept_label']}' "
                      f"→ Lore (ID: {lore_id})")

    if verbose and not cycle_log['promoted']:
        print("[SongKeeper] No landmarks ready for promotion this cycle.")

    # ------------------------------------------------------------------
    # STEP 6 — Write Dream Diary
    # ------------------------------------------------------------------
    _write_dream_diary(cycle_log)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[SongKeeper] Cycle complete — {now_str}")
        print(f"  Decayed:  {cycle_log['edges_decayed']} edges")
        print(f"  Pruned:   {cycle_log['edges_pruned']} edges")
        print(f"  Updated:  {len(cycle_log['relevancy_updates'])} relevancy scores")
        print(f"  Promoted: {len(cycle_log['promoted'])} landmarks to Lore\n")

    return cycle_log


# ---------------------------------------------------------------------------
# CRON ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cycle(verbose=True)
