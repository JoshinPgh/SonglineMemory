"""
core_memory.py — SonglineMemory Core Database Layer
JSG Labs / Geldrich Corp

The foundation. All tables, all schema, all base read/write operations.
Nothing gets built without this being right first.

Architecture:
    LANDMARKS       — Active working memory. Compressed semantic nodes.
    SONGLINES       — Narrative edges connecting Landmarks. Weighted by use.
    LORE            — Long-term graduated memory. The wiki layer.
                      Landmarks that prove themselves across multiple contexts
                      migrate here via the SongKeeper cycle.
    DREAM_DIARY     — Human-readable log of every SongKeeper decision.
                      Trust through transparency.

SongKeeper Promotion Gate (a Landmark graduates to LORE only when ALL three pass):
    1. relevancy_score   >= RELEVANCY_THRESHOLD
    2. recall_count      >= MIN_RECALL_COUNT
    3. unique_query_count >= MIN_UNIQUE_QUERY_COUNT

Zero external dependencies. Pure Python stdlib.
"""

import sqlite3
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DB_NAME = "/mnt/SonglineMemory/landscape.db"

# SongKeeper promotion gate thresholds — adjust as the system matures
RELEVANCY_THRESHOLD    = 0.60   # 0.0–1.0 score
MIN_RECALL_COUNT       = 3      # must have been recalled at least N times
MIN_UNIQUE_QUERY_COUNT = 2      # must have been reached from at least N distinct queries


# ---------------------------------------------------------------------------
# SCHEMA INIT
# ---------------------------------------------------------------------------

def init_db():
    """
    Initializes the SQLite database and all tables.
    Safe to run on an existing database — uses IF NOT EXISTS throughout.
    Adds missing columns to existing tables via ALTER TABLE where needed.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # --- LANDMARKS ---
    # Active working memory. Short and medium term.
    # Every Landmark starts here. SongKeeper decides if it graduates to Lore.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS landmarks (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_label       TEXT    NOT NULL,
        core_data           TEXT    NOT NULL,
        creation_date       TEXT    NOT NULL,
        last_accessed       TEXT,
        relevancy_score     REAL    NOT NULL DEFAULT 0.5,
        recall_count        INTEGER NOT NULL DEFAULT 0,
        unique_query_count  INTEGER NOT NULL DEFAULT 0,
        memory_tier         TEXT    NOT NULL DEFAULT 'active'
        -- memory_tier values: 'active' | 'lore' | 'archived'
    )
    ''')

    # --- SONGLINES ---
    # Narrative edges between Landmarks.
    # strength_score and walk_count are the SongKeeper's instruments —
    # heavily walked edges strengthen, neglected edges decay.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS songlines (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        origin_id           INTEGER NOT NULL,
        destination_id      INTEGER NOT NULL,
        narrative_context   TEXT    NOT NULL,
        strength_score      REAL    NOT NULL DEFAULT 0.5,
        walk_count          INTEGER NOT NULL DEFAULT 0,
        creation_date       TEXT    NOT NULL,
        last_walked         TEXT,
        FOREIGN KEY(origin_id)      REFERENCES landmarks(id),
        FOREIGN KEY(destination_id) REFERENCES landmarks(id)
    )
    ''')

    # --- LORE ---
    # Long-term graduated memory. The wiki layer.
    # Landmarks that pass the SongKeeper's three-gate promotion test
    # are migrated here. Nothing is deleted — it changes address.
    # The Lore grows richer over time and never forgets.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lore (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        origin_landmark_id  INTEGER NOT NULL,   -- original landmarks.id for traceability
        concept_label       TEXT    NOT NULL,
        core_data           TEXT    NOT NULL,
        original_created    TEXT    NOT NULL,   -- when the Landmark was first placed
        graduated_date      TEXT    NOT NULL,   -- when SongKeeper moved it here
        relevancy_score     REAL    NOT NULL,   -- score at time of graduation
        recall_count        INTEGER NOT NULL,
        unique_query_count  INTEGER NOT NULL,
        songkeeper_notes    TEXT                -- why it was graduated (from Dream Diary)
    )
    ''')

    # --- DREAM_DIARY ---
    # Every SongKeeper cycle writes here.
    # Human-readable. Append-only. The conscience of the system.
    # action values: 'promoted' | 'archived' | 'strengthened' | 'weakened' |
    #                'merged' | 'flagged' | 'survived'
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dream_diary (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_date      TEXT    NOT NULL,
        action          TEXT    NOT NULL,
        target_type     TEXT    NOT NULL,   -- 'landmark' | 'songline'
        target_id       INTEGER NOT NULL,
        concept_label   TEXT,
        reason          TEXT    NOT NULL,
        score_snapshot  TEXT                -- JSON snapshot of scores at decision time
    )
    ''')

    conn.commit()
    conn.close()
    print("SUCCESS: SonglineMemory landscape initialized.")
    print(f"  DB: {DB_NAME}")
    print(f"  Promotion gate: relevancy>={RELEVANCY_THRESHOLD} | "
          f"recalls>={MIN_RECALL_COUNT} | unique_queries>={MIN_UNIQUE_QUERY_COUNT}")


# ---------------------------------------------------------------------------
# LANDMARK OPERATIONS
# ---------------------------------------------------------------------------

def add_landmark(concept: str, data: str) -> int:
    """
    Places a new Landmark in active memory.
    Returns the new landmark ID.
    Feeds directly from Compressor.compress() output.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO landmarks
        (concept_label, core_data, creation_date, last_accessed,
         relevancy_score, recall_count, unique_query_count, memory_tier)
    VALUES (?, ?, ?, ?, 0.5, 0, 0, 'active')
    ''', (concept, data, now, now))

    landmark_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"LANDMARK PLACED: '{concept}' (ID: {landmark_id})")
    return landmark_id


def recall_landmark(landmark_id: int, query_signature: str = None) -> dict | None:
    """
    Retrieves a Landmark by ID and increments its recall_count.
    If query_signature is provided and is new for this landmark,
    increments unique_query_count as well.

    Returns the landmark as a dict, or None if not found.

    Note: unique_query_count tracking is simplified here —
    full deduplication requires a query_log table (Phase 2).
    For now, caller passes a signature and we trust it.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM landmarks WHERE id = ?', (landmark_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Always increment recall_count
    # Increment unique_query_count only when a new query_signature is provided
    # (simplified — full dedup in Phase 2)
    if query_signature:
        cursor.execute('''
        UPDATE landmarks
        SET recall_count = recall_count + 1,
            unique_query_count = unique_query_count + 1,
            last_accessed = ?
        WHERE id = ?
        ''', (now, landmark_id))
    else:
        cursor.execute('''
        UPDATE landmarks
        SET recall_count = recall_count + 1,
            last_accessed = ?
        WHERE id = ?
        ''', (now, landmark_id))

    conn.commit()

    # Re-fetch updated row
    cursor.execute('SELECT * FROM landmarks WHERE id = ?', (landmark_id,))
    updated = dict(cursor.fetchone())
    conn.close()

    return updated


def update_relevancy(landmark_id: int, new_score: float) -> None:
    """
    Updates the relevancy_score for a Landmark.
    Called by SongKeeper during the consolidation cycle.
    Score is clamped to 0.0–1.0.
    """
    score = max(0.0, min(1.0, new_score))
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE landmarks SET relevancy_score = ? WHERE id = ?',
        (score, landmark_id)
    )
    conn.commit()
    conn.close()


def get_active_landmarks() -> list[dict]:
    """Returns all Landmarks in active memory tier."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM landmarks WHERE memory_tier = 'active' ORDER BY creation_date DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# SONGLINE OPERATIONS
# ---------------------------------------------------------------------------

def add_songline(origin_id: int, destination_id: int, narrative: str) -> int:
    """
    Weaves a new Songline edge between two Landmarks.
    Returns the new songline ID.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO songlines
        (origin_id, destination_id, narrative_context,
         strength_score, walk_count, creation_date, last_walked)
    VALUES (?, ?, ?, 0.5, 0, ?, NULL)
    ''', (origin_id, destination_id, narrative, now))

    songline_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"SONGLINE WOVEN: {origin_id} → {destination_id} (ID: {songline_id})")
    return songline_id


def walk_songline(songline_id: int) -> None:
    """
    Records a traversal of a Songline edge.
    Increments walk_count and updates last_walked.
    SongKeeper uses walk_count to strengthen or weaken edges.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    UPDATE songlines
    SET walk_count = walk_count + 1,
        last_walked = ?
    WHERE id = ?
    ''', (now, songline_id))

    conn.commit()
    conn.close()


def update_songline_strength(songline_id: int, new_strength: float) -> None:
    """
    Updates the strength_score of a Songline edge.
    Called by SongKeeper. Clamped to 0.0–1.0.
    """
    strength = max(0.0, min(1.0, new_strength))
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE songlines SET strength_score = ? WHERE id = ?',
        (strength, songline_id)
    )
    conn.commit()
    conn.close()


def get_songlines_for_landmark(landmark_id: int) -> list[dict]:
    """
    Returns all Songlines connected to a given Landmark
    (as origin or destination), ordered by strength descending.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM songlines
    WHERE origin_id = ? OR destination_id = ?
    ORDER BY strength_score DESC
    ''', (landmark_id, landmark_id))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# LORE OPERATIONS
# ---------------------------------------------------------------------------

def graduate_to_lore(landmark_id: int, notes: str = "") -> int | None:
    """
    Migrates a Landmark from active memory to the Lore.
    Called by SongKeeper when a Landmark passes all three promotion gates.

    Returns the new Lore entry ID, or None if the Landmark doesn't qualify
    or doesn't exist.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM landmarks WHERE id = ? AND memory_tier = 'active'", (landmark_id,))
    lm = cursor.fetchone()

    if not lm:
        conn.close()
        return None

    lm = dict(lm)

    # Verify all three gates
    qualifies = (
        lm['relevancy_score']    >= RELEVANCY_THRESHOLD and
        lm['recall_count']       >= MIN_RECALL_COUNT and
        lm['unique_query_count'] >= MIN_UNIQUE_QUERY_COUNT
    )

    if not qualifies:
        conn.close()
        return None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert into Lore
    cursor.execute('''
    INSERT INTO lore
        (origin_landmark_id, concept_label, core_data, original_created,
         graduated_date, relevancy_score, recall_count, unique_query_count, songkeeper_notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        lm['id'], lm['concept_label'], lm['core_data'], lm['creation_date'],
        now, lm['relevancy_score'], lm['recall_count'], lm['unique_query_count'], notes
    ))

    lore_id = cursor.lastrowid

    # Update the Landmark's tier — keep it in landmarks table for graph integrity
    # but mark it as 'lore' so active queries skip it
    cursor.execute(
        "UPDATE landmarks SET memory_tier = 'lore' WHERE id = ?",
        (landmark_id,)
    )

    conn.commit()
    conn.close()

    print(f"GRADUATED TO LORE: '{lm['concept_label']}' (Landmark {landmark_id} → Lore {lore_id})")
    return lore_id


def get_lore(search_term: str = None) -> list[dict]:
    """
    Returns Lore entries, optionally filtered by search_term
    matching concept_label or core_data.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if search_term:
        pattern = f"%{search_term}%"
        cursor.execute('''
        SELECT * FROM lore
        WHERE concept_label LIKE ? OR core_data LIKE ?
        ORDER BY graduated_date DESC
        ''', (pattern, pattern))
    else:
        cursor.execute('SELECT * FROM lore ORDER BY graduated_date DESC')

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# DREAM DIARY OPERATIONS
# ---------------------------------------------------------------------------

def log_dream(action: str, target_type: str, target_id: int,
              concept_label: str = "", reason: str = "",
              score_snapshot: str = "") -> None:
    """
    Writes a single entry to the Dream Diary.
    Called by SongKeeper for every decision it makes.

    action       : 'promoted' | 'archived' | 'strengthened' | 'weakened' |
                   'merged' | 'flagged' | 'survived'
    target_type  : 'landmark' | 'songline'
    score_snapshot: JSON string of relevant scores at decision time
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO dream_diary
        (cycle_date, action, target_type, target_id,
         concept_label, reason, score_snapshot)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (now, action, target_type, target_id, concept_label, reason, score_snapshot))

    conn.commit()
    conn.close()


def get_dream_diary(limit: int = 50) -> list[dict]:
    """Returns the most recent Dream Diary entries."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM dream_diary
    ORDER BY cycle_date DESC
    LIMIT ?
    ''', (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def export_dream_diary_md(filepath: str = "DREAMS.md") -> None:
    """
    Exports the Dream Diary to a human-readable markdown file.
    This is the transparency layer — reviewable by anyone without
    touching the database.
    """
    entries = get_dream_diary(limit=500)

    lines = [
        "# SonglineMemory — Dream Diary",
        f"*Generated by SongKeeper on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "---",
        ""
    ]

    if not entries:
        lines.append("*No SongKeeper cycles have run yet.*")
    else:
        current_date = None
        for e in entries:
            date_str = e['cycle_date'][:10]
            if date_str != current_date:
                current_date = date_str
                lines.append(f"## {date_str}")
                lines.append("")

            icon = {
                'promoted':    '⬆️',
                'archived':    '📦',
                'strengthened':'💪',
                'weakened':    '📉',
                'merged':      '🔗',
                'flagged':     '🚩',
                'survived':    '✅',
            }.get(e['action'], '•')

            lines.append(
                f"{icon} **{e['action'].upper()}** | "
                f"{e['target_type']} #{e['target_id']} | "
                f"*{e['concept_label']}*"
            )
            lines.append(f"   > {e['reason']}")
            if e['score_snapshot']:
                lines.append(f"   > Scores: `{e['score_snapshot']}`")
            lines.append("")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"DREAM DIARY EXPORTED: {filepath} ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# BOOTSTRAP
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Booting SonglineMemory landscape...")
    init_db()
    print()
    print("Schema ready. Tables: landmarks, songlines, lore, dream_diary")
    print("Next: build retrieval function and auto-Songline weave on write.")
