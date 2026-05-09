"""
retrieval.py — SonglineMemory Retrieval Engine
JSG Labs / Geldrich Corp

Phase 1 & 2.

Walks the landscape from a query. Finds relevant Landmarks via TF-IDF
cosine similarity, then traverses their Songline edges to return the
full narrative path — not just a fact, but the context of how it connects.

This is the read side of the pipeline:
    raw query → TF-IDF score against Landmarks → ranked matches
              → walk connected Songlines → return narrative path

Also supports direct Lore queries — search the long-term knowledge base
independently of the active graph.

Zero external dependencies. Pure Python + math.
Same TF-IDF logic used in compressor.py — consistent scoring
on both write and read ends of the system.

Public interface:
    query(text, top_n)          — main entry point, searches active landscape
    walk_path(landmark_id)      — full Songline narrative for one Landmark
    search_lore(text, tag)      — search the Lore knowledge base directly
    format_results(results)     — formats results for LLM prompt injection
"""

import math
import string
import sqlite3

from core_memory import (
    DB_NAME,
    recall_landmark,
    walk_songline,
    get_songlines_for_landmark,
    query_lore,
)


# ---------------------------------------------------------------------------
# STOP WORDS — kept local to avoid circular import with compressor
# ---------------------------------------------------------------------------

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'we',
    'you', 'he', 'she', 'they', 'my', 'our', 'your', 'their',
    'and', 'or', 'but', 'so', 'if', 'as', 'not', 'no', 'nor',
    'about', 'up', 'out', 'what', 'which', 'who', 'when', 'where',
    'there', 'here', 'just', 'also', 'then', 'than', 'into', 'go',
}


# ---------------------------------------------------------------------------
# TF-IDF HELPERS
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words, min length 2."""
    return [
        w for w in
        text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        if w not in STOP_WORDS and len(w) > 2
    ]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Term frequency — count normalized by document length."""
    if not tokens:
        return {}
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def _build_tfidf_vector(tokens: list[str],
                         idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF vector for a document given a pre-computed IDF table."""
    tf = _term_freq(tokens)
    return {t: tf_val * idf.get(t, 0.0) for t, tf_val in tf.items()}


def _cosine_similarity(vec_a: dict[str, float],
                        vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    dot   = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vec_b)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _compute_idf(corpus: list[list[str]]) -> dict[str, float]:
    """
    Inverse document frequency across the corpus.
    IDF = log((1 + N) / (1 + df)) + 1 — smoothed to avoid zero division.
    """
    N  = len(corpus)
    df = {}
    for doc_tokens in corpus:
        for term in set(doc_tokens):
            df[term] = df.get(term, 0) + 1
    return {
        term: math.log((1 + N) / (1 + count)) + 1
        for term, count in df.items()
    }


# ---------------------------------------------------------------------------
# LANDMARK SCORING
# ---------------------------------------------------------------------------

def _load_active_landmarks() -> list[dict]:
    """Loads all active Landmarks from landscape.db."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, concept_label, core_data, relevancy_score,
               recall_count, unique_query_count, last_accessed
        FROM landmarks
        WHERE memory_tier = 'active'
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def _score_landmarks(query_text: str,
                     landmarks: list[dict]) -> list[tuple[float, dict]]:
    """
    Scores each active Landmark against the query using TF-IDF cosine similarity.

    Searchable text = concept_label + core_data combined.
    Relevancy score gives proven memories a mild boost (max 20%).

    Returns list of (score, landmark_dict) sorted descending.
    """
    if not landmarks:
        return []

    landmark_texts       = [f"{lm['concept_label']} {lm['core_data']}" for lm in landmarks]
    landmark_token_lists = [_tokenize(t) for t in landmark_texts]
    query_tokens         = _tokenize(query_text)
    full_corpus          = landmark_token_lists + [query_tokens]
    idf                  = _compute_idf(full_corpus)
    query_vec            = _build_tfidf_vector(query_tokens, idf)

    scored = []
    for lm, tokens in zip(landmarks, landmark_token_lists):
        lm_vec            = _build_tfidf_vector(tokens, idf)
        similarity        = _cosine_similarity(query_vec, lm_vec)
        relevancy_boost   = 1.0 + (lm['relevancy_score'] * 0.2)
        final_score       = similarity * relevancy_boost
        if final_score > 0.0:
            scored.append((final_score, lm))

    return sorted(scored, reverse=True)


# ---------------------------------------------------------------------------
# SONGLINE PATH BUILDER
# ---------------------------------------------------------------------------

def _build_narrative_path(landmark: dict,
                           max_edges: int = 5) -> dict:
    """
    Retrieves strongest Songlines connected to a Landmark.
    Records a walk on each traversed edge (increments walk_count + strength).
    Returns a result dict ready for the caller.
    """
    songlines = get_songlines_for_landmark(landmark['id'])
    top_edges = sorted(
        songlines, key=lambda s: s['strength_score'], reverse=True
    )[:max_edges]

    for edge in top_edges:
        walk_songline(edge['id'])

    narrative_fragments = []
    for edge in top_edges:
        direction = "→" if edge['origin_id'] == landmark['id'] else "←"
        narrative_fragments.append({
            'songline_id':  edge['id'],
            'direction':    direction,
            'connected_to': edge['destination_id'] if direction == "→" else edge['origin_id'],
            'narrative':    edge['narrative_context'],
            'strength':     edge['strength_score'],
            'walk_count':   edge['walk_count'],
        })

    return {
        'landmark_id':    landmark['id'],
        'concept_label':  landmark['concept_label'],
        'core_data':      landmark['core_data'],
        'relevancy_score': landmark['relevancy_score'],
        'recall_count':   landmark.get('recall_count', 0),
        'songlines':      narrative_fragments,
    }


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE — ACTIVE LANDSCAPE
# ---------------------------------------------------------------------------

def query(text: str,
          top_n: int = 5,
          query_signature: str = None) -> list[dict]:
    """
    Main entry point. Queries the active SonglineMemory landscape.

    Returns up to top_n results, each containing:
        - Matched Landmark (label, core_data, relevancy_score, recall_count)
        - Strongest connected Songlines (narrative path)
        - Similarity score that ranked it

    query_signature: optional string identifying this query context.
    Used to increment unique_query_count. Defaults to query text itself.

    Side effects:
        - Increments recall_count on matched Landmarks
        - Increments unique_query_count when query_signature is new
        - Increments walk_count + strength on traversed Songline edges

    Example:
        results = query("TF-IDF compression encoder")
        print(format_results(results))
    """
    if not text or not text.strip():
        return []

    sig       = query_signature or text.strip()
    landmarks = _load_active_landmarks()

    if not landmarks:
        return []

    scored = _score_landmarks(text, landmarks)
    top    = scored[:top_n]

    results = []
    for score, lm in top:
        updated_lm = recall_landmark(lm['id'], query_signature=sig)
        if updated_lm:
            lm.update(updated_lm)
        path          = _build_narrative_path(lm)
        path['score'] = round(score, 4)
        results.append(path)

    return results


def walk_path(landmark_id: int) -> dict | None:
    """
    Direct path walk from a known Landmark ID.
    Useful when the caller already knows which Landmark they want.
    Returns the narrative path dict or None if not found.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, concept_label, core_data, relevancy_score,
                  recall_count, unique_query_count, last_accessed
           FROM landmarks WHERE id = ?""",
        (landmark_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    lm = dict(row)
    recall_landmark(landmark_id)
    return _build_narrative_path(lm)


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE — LORE
# ---------------------------------------------------------------------------

def search_lore(text: str = None,
                tag: str = None,
                limit: int = 20) -> list[dict]:
    """
    Search the Lore knowledge base directly.

    Lore is the long-term wiki layer — memories that passed all three
    SongKeeper promotion gates and were graduated from the active graph.
    Independently queryable for company KB, NotebookLM ingestion, etc.

    text : free-text search against concept_label and core_data
    tag  : filter by tag string (e.g. source tag from batch import)
    limit: max results to return (default 20)

    Example:
        # Search by topic
        results = search_lore("TF-IDF encoder")

        # Search by import source
        results = search_lore(tag="gemini_wiki")
    """
    return query_lore(search_text=text, tag=tag, limit=limit)


def format_lore_results(results: list[dict]) -> str:
    """Formats Lore search results for human or LLM reading."""
    if not results:
        return "[SonglineMemory Lore] No entries found."

    lines = ["[SonglineMemory — Lore]", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"[LORE {i}]")
        lines.append(f"CONCEPT: {r['concept_label']}")
        lines.append(f"FACT: {r['core_data']}")
        lines.append(f"PROMOTED: {r['promoted_date']}")
        lines.append(f"RECALLS: {r['recall_count']} | "
                     f"UNIQUE QUERIES: {r['unique_query_count']} | "
                     f"RELEVANCY: {r['relevancy_score']:.2f}")
        if r.get('tags') and r['tags'] != '[]':
            lines.append(f"TAGS: {r['tags']}")
        lines.append("")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# FORMAT FOR LLM INJECTION
# ---------------------------------------------------------------------------

def format_results(results: list[dict]) -> str:
    """
    Formats active landscape query results for LLM prompt injection.
    This is how retrieved memory gets surfaced to the model.

    Example output:
        [MEMORY 1 | score: 0.8432]
        LANDMARK: TF-IDF Encoder Design
        FACT: Encoder uses TF-IDF weighted cosine similarity...
        SONGLINES:
          → "The encoder decision led to replacing the hash compressor..." (strength: 0.72)
    """
    if not results:
        return "[SonglineMemory] No relevant memories found."

    lines = ["[SonglineMemory — Retrieved Context]", ""]

    for i, r in enumerate(results, 1):
        lines.append(f"[MEMORY {i} | score: {r['score']}]")
        lines.append(f"LANDMARK: {r['concept_label']}")
        lines.append(f"FACT: {r['core_data']}")
        lines.append(f"RELEVANCY: {r['relevancy_score']:.2f} | "
                     f"RECALLS: {r['recall_count']}")

        if r['songlines']:
            lines.append("SONGLINES:")
            for s in r['songlines']:
                lines.append(
                    f"  {s['direction']} \"{s['narrative']}\" "
                    f"(strength: {s['strength']:.2f} | walked: {s['walk_count']}x)"
                )
        else:
            lines.append("SONGLINES: none yet")

        lines.append("")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUERY = "TF-IDF compression memory encoder"

    print("=== RETRIEVAL TEST ===\n")
    print(f"QUERY: {TEST_QUERY}\n")

    results = query(TEST_QUERY, top_n=3)

    if not results:
        print("No results. Add some Landmarks first via pipeline.remember()")
    else:
        print(format_results(results))

    print(f"{len(results)} result(s) returned.")

    print("\n=== LORE TEST ===\n")
    lore = search_lore("memory")
    if not lore:
        print("Lore is empty — nothing promoted yet. SongKeeper will populate it over time.")
    else:
        print(format_lore_results(lore))
