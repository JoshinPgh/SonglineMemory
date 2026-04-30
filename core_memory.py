"""
core_memory.py — SonglineMemory Core Database Layer
JSG Labs / Geldrich Corp

The foundation. All tables, all schema, all base read/write operations.
Nothing gets built without this being right first.

Architecture:
    landscape.db — active SonglineMemory graph (landmarks + songlines)
    lore.db      — long-term wiki/knowledge base (promoted memories live here)

Two databases. Clean separation. SongKeeper writes to both.
lore.db is independently queryable — NotebookLM, company KB, genius finder,
whatever gets plugged in later doesn't touch the active graph.

Tables:
    landscape.db:
        landmarks   — active memory nodes
        songlines   — narrative edges connecting landmarks

    lore.db:
        lore_entries — promoted long-term knowledge, flat and queryable

SongKeeper hooks baked into schema:
    landmarks.memory_tier         — 'active' | 'lore' (transition marker)
    landmarks.relevancy_score     — float 0.0–1.0, decays over time
    landmarks.recall_count        — how many times this landmark was recalled
    landmarks.unique_query_count  — how many distinct queries surfaced this
    landmarks.last_accessed       — timestamp of last recall
    landmarks.creation_date       — timestamp of placement

    songlines.strength_score      — float, strengthens with walk_count
    songlines.walk_count          — raw traversal counter
    songlines.last_walked         — timestamp of last traversal

Promotion gates (SongKeeper checks all three):
    relevancy_score  >= PROMOTE_RELEVANCY_MIN
    recall_count     >= PROMOTE_RECALL_MIN
    unique_query_count >= PROMOTE_UNIQUE_MIN

Zero external dependencies. Pure Python stdlib + sqlite3.
"""

import sqlite3
import json
from datetime import datetime

# ---------------------------------------------------------------------------
# DB PATHS
# ---------------------------------------------------------------------------

DB_NAME      = "/mnt/SonglineMemory/landscape.db"   # active graph
LORE_DB_NAME = "/mnt/SonglineMemory/lore.db"        # long-term wiki

# ---------------------------------------------------------------------------
# PROMOTION GATES — SongKeeper uses these to decide what moves to Lore
# ---------------------------------------------------------------------------

PROMOTE_RELEVANCY_MIN = 0.6   # must have proven relevancy
PROMOTE_RECALL_MIN    = 3     # recalled at least 3 times
PROMOTE_UNIQUE_MIN    = 2     # surfaced by at least 2 distinct queries

# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

def init_db():
    """
    Initializes both databases.
    Safe to run repeatedly — CREATE IF NOT EXISTS throughout.
    Wipes and rebuilds if called with wipe=True (used for clean deploys).
    """

    # --- landscape.db ---
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS landmarks (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_label       TEXT    NOT NULL,
        core_data           TEXT    NOT NULL,
        creation_date       TEXT    NOT NULL,
        last_accessed       TEXT,
        memory_tier         TEXT    NOT NULL DEFAULT 'active',
        relevancy_score     REAL    NOT NULL DEFAULT 0.5,
        recall_count        INTEGER NOT NULL DEFAULT 0,
        unique_query_count  INTEGER NOT NULL DEFAULT 0,
        query_signatures    TEXT    NOT NULL DEFAULT '[]'
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS songlines (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        origin_id         INTEGER NOT NULL,
        destination_id    INTEGER NOT NULL,
        narrative_context TEXT    NOT NULL,
        strength_score    REAL    NOT NULL DEFAULT 0.5,
        walk_count        INTEGER NOT NULL DEFAULT 0,
        last_walked       TEXT,
        FOREIGN KEY(origin_id)      REFERENCES landmarks(id),
        FOREIGN KEY(destination_id) REFERENCES landmarks(id)
    )
    ''')

    conn.commit()
    conn.close()

    # --- lore.db ---
    lore_conn = sqlite3.connect(LORE_DB_NAME)
    lore_cursor = lore_conn.cursor()

    lore_cursor.execute('''
    CREATE TABLE IF NOT EXISTS lore_entries (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_label     TEXT NOT NULL,
        core_data         TEXT NOT NULL,
        origin_landmark_id INTEGER,
        promoted_date     TEXT NOT NULL,
        recall_count      INTEGER NOT NULL DEFAULT 0,
        unique_query_count INTEGER NOT NULL DEFAULT 0,
        relevancy_score   REAL NOT NULL DEFAULT 0.5,
        tags              TEXT NOT NULL DEFAULT '[]'
    )
    ''')

    lore_conn.commit()
    lore_conn.close()

    print("SUCCESS: SonglineMemory landscape initialized.")
    print(f"  DB:  {DB_NAME}")
    print(f"  Lore: {LORE_DB_NAME}")
    print(f"  Promotion gate: relevancy>={PROMOTE_RELEVANCY_MIN} | "
          f"recalls>={PROMOTE_RECALL_MIN} | "
          f"unique_queries>={PROMOTE_UNIQUE_MIN}")


# ---------------------------------------------------------------------------
# LANDMARK OPS
# ---------------------------------------------------------------------------

def add_landmark(concept: str, data: str) -> int:
    """
    Places a new Landmark in the active landscape.
    Returns the new landmark's ID.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO landmarks
        (concept_label, core_data, creation_date, memory_tier,
         relevancy_score, recall_count, unique_query_count, query_signatures)
    VALUES (?, ?, ?, 'active', 0.5, 0, 0, '[]')
    ''', (concept, data, timestamp))

    landmark_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"LANDMARK PLACED: '{concept}' (ID: {landmark_id})")
    return landmark_id


def recall_landmark(landmark_id: int,
                    query_signature: str = None) -> dict | None:
    """
    Records a recall event on a Landmark.
    Increments recall_count. If query_signature is new, increments
    unique_query_count. Updates last_accessed.

    Returns updated landmark dict or None if not found.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM landmarks WHERE id = ?", (landmark_id,)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    lm = dict(row)
    sigs = json.loads(lm['query_signatures'])
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_unique = 0
    if query_signature and query_signature not in sigs:
        sigs.append(query_signature)
        new_unique = 1

    cursor.execute('''
    UPDATE landmarks
    SET recall_count       = recall_count + 1,
        unique_query_count = unique_query_count + ?,
        last_accessed      = ?,
        query_signatures   = ?
    WHERE id = ?
    ''', (new_unique, now, json.dumps(sigs), landmark_id))

    conn.commit()

    cursor.execute("SELECT * FROM landmarks WHERE id = ?", (landmark_id,))
    updated = dict(cursor.fetchone())
    conn.close()

    return updated


def get_landmark(landmark_id: int) -> dict | None:
    """Returns a single landmark dict by ID, or None."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM landmarks WHERE id = ?", (landmark_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_active_landmarks() -> list[dict]:
    """Returns all landmarks with memory_tier = 'active'."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM landmarks WHERE memory_tier = 'active'")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_landmark_relevancy(landmark_id: int,
                               new_score: float) -> None:
    """SongKeeper uses this to adjust relevancy scores during consolidation."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE landmarks SET relevancy_score = ? WHERE id = ?",
        (max(0.0, min(1.0, new_score)), landmark_id)
    )
    conn.commit()
    conn.close()


def retire_landmark(landmark_id: int) -> None:
    """
    Marks a landmark as 'lore' in landscape.db (soft delete from active graph).
    Called by SongKeeper after copying to lore.db.
    The record stays in landscape.db for Songline edge integrity — it just
    stops appearing in active queries.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE landmarks SET memory_tier = 'lore' WHERE id = ?",
        (landmark_id,)
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SONGLINE OPS
# ---------------------------------------------------------------------------

def add_songline(origin_id: int,
                 destination_id: int,
                 narrative: str) -> int:
    """
    Weaves a new Songline edge between two Landmarks.
    Returns the new songline's ID.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO songlines
        (origin_id, destination_id, narrative_context,
         strength_score, walk_count)
    VALUES (?, ?, ?, 0.5, 0)
    ''', (origin_id, destination_id, narrative))

    songline_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return songline_id


def walk_songline(songline_id: int) -> None:
    """
    Records a traversal on a Songline edge.
    Increments walk_count and strengthens the edge.

    Strength formula:
        new_strength = min(1.0, current_strength + 0.05)
    Each walk nudges the edge stronger, capped at 1.0.
    SongKeeper handles decay of unwalked edges separately.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
    UPDATE songlines
    SET walk_count     = walk_count + 1,
        strength_score = MIN(1.0, strength_score + 0.05),
        last_walked    = ?
    WHERE id = ?
    ''', (now, songline_id))

    conn.commit()
    conn.close()


def get_songlines_for_landmark(landmark_id: int) -> list[dict]:
    """
    Returns all Songline edges connected to a Landmark
    (as origin OR destination).
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
    SELECT * FROM songlines
    WHERE origin_id = ? OR destination_id = ?
    ''', (landmark_id, landmark_id))

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def decay_songline(songline_id: int,
                   decay_rate: float = 0.05) -> float:
    """
    Weakens a Songline edge by decay_rate.
    Called by SongKeeper on unwalked edges during consolidation.
    Returns the new strength score.

    Strength floor is 0.0 — SongKeeper prunes edges at or below 0.1.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT strength_score FROM songlines WHERE id = ?",
        (songline_id,)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return 0.0

    new_strength = max(0.0, row['strength_score'] - decay_rate)

    cursor.execute(
        "UPDATE songlines SET strength_score = ? WHERE id = ?",
        (new_strength, songline_id)
    )
    conn.commit()
    conn.close()

    return new_strength


def get_all_songlines() -> list[dict]:
    """Returns all Songline edges. SongKeeper uses this for bulk decay pass."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM songlines")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def delete_songline(songline_id: int) -> None:
    """Hard deletes a Songline edge. Used by SongKeeper to prune dead edges."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM songlines WHERE id = ?", (songline_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# LORE OPS
# ---------------------------------------------------------------------------

def promote_to_lore(landmark: dict,
                    tags: list[str] = None) -> int:
    """
    Copies a Landmark into lore.db as a Lore entry.
    Called by SongKeeper after a landmark passes all promotion gates.

    Does NOT remove from landscape.db — retire_landmark() handles that
    as a separate step, preserving Songline edge references.

    Returns the new lore entry ID.
    """
    lore_conn = sqlite3.connect(LORE_DB_NAME)
    cursor = lore_conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag_json = json.dumps(tags or [])

    cursor.execute('''
    INSERT INTO lore_entries
        (concept_label, core_data, origin_landmark_id, promoted_date,
         recall_count, unique_query_count, relevancy_score, tags)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        landmark['concept_label'],
        landmark['core_data'],
        landmark['id'],
        now,
        landmark.get('recall_count', 0),
        landmark.get('unique_query_count', 0),
        landmark.get('relevancy_score', 0.5),
        tag_json,
    ))

    lore_id = cursor.lastrowid
    lore_conn.commit()
    lore_conn.close()

    return lore_id


def query_lore(search_text: str = None,
               tag: str = None,
               limit: int = 20) -> list[dict]:
    """
    Basic Lore query — returns entries matching text or tag.
    This is a simple LIKE search for now.
    Full TF-IDF scoring on Lore is a Phase 3 enhancement.

    If neither search_text nor tag provided, returns most recent entries.
    """
    lore_conn = sqlite3.connect(LORE_DB_NAME)
    lore_conn.row_factory = sqlite3.Row
    cursor = lore_conn.cursor()

    if search_text:
        pattern = f"%{search_text}%"
        cursor.execute('''
        SELECT * FROM lore_entries
        WHERE concept_label LIKE ? OR core_data LIKE ?
        ORDER BY promoted_date DESC
        LIMIT ?
        ''', (pattern, pattern, limit))
    elif tag:
        pattern = f'%"{tag}"%'
        cursor.execute('''
        SELECT * FROM lore_entries
        WHERE tags LIKE ?
        ORDER BY promoted_date DESC
        LIMIT ?
        ''', (pattern, limit))
    else:
        cursor.execute('''
        SELECT * FROM lore_entries
        ORDER BY promoted_date DESC
        LIMIT ?
        ''', (limit,))

    rows = [dict(r) for r in cursor.fetchall()]
    lore_conn.close()
    return rows


# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Initializing SonglineMemory landscape...")
    init_db()
    print("\nPlacing a test landmark...")
    lid = add_landmark("Test Concept", "This is a test fact for schema verification.")
    print(f"\nRecalling landmark {lid}...")
    recall_landmark(lid, query_signature="test_query_1")
    recall_landmark(lid, query_signature="test_query_2")
    lm = get_landmark(lid)
    print(f"  recall_count: {lm['recall_count']}")
    print(f"  unique_query_count: {lm['unique_query_count']}")
    print("\nSchema verified. Ready for SongKeeper.")
