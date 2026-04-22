"""
retrieval.py — SonglineMemory Retrieval Engine
JSG Labs / Geldrich Corp

Phase 1, Item 2.

Walks the landscape from a query. Finds relevant Landmarks via TF-IDF
cosine similarity, then traverses their Songline edges to return the
full narrative path — not just a fact, but the context of how it connects.

This is the read side of the pipeline:
    raw query → TF-IDF score against Landmarks → ranked matches
              → walk connected Songlines → return narrative path

Zero external dependencies. Pure Python + math.
Same TF-IDF logic used in Compressor.py — consistent scoring
on both write and read ends of the system.

Public interface:
    query(text, top_n)     — main entry point, returns ranked results
    walk_path(landmark_id) — returns full Songline narrative for one Landmark
"""

import math
import string
import sqlite3
from datetime import datetime

from core_memory import (
    DB_NAME,
    recall_landmark,
    walk_songline,
    get_songlines_for_landmark,
)


# ---------------------------------------------------------------------------
# STOP WORDS — shared with Compressor, kept local to avoid circular import
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


def _build_tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF vector for a document given a pre-computed IDF table."""
    tf = _term_freq(tokens)
    return {t: tf_val * idf.get(t, 0.0) for t, tf_val in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vec_b)

    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _compute_idf(corpus: list[list[str]]) -> dict[str, float]:
    """
    Compute inverse document frequency across the corpus.
    IDF = log((1 + N) / (1 + df)) + 1  — smoothed to avoid zero division.
    """
    N = len(corpus)
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
    """
    Loads all active Landmarks from the database.
    Returns list of dicts with id, concept_label, core_data, relevancy_score.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, concept_label, core_data, relevancy_score
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

    The Landmark's searchable text is: concept_label + " " + core_data
    Relevancy score is used as a mild multiplier — proven memories surface
    slightly ahead of equally similar but untested ones.

    Returns list of (score, landmark_dict) sorted descending.
    """
    if not landmarks:
        return []

    # Build corpus: one document per landmark (label + data combined)
    landmark_texts = [
        f"{lm['concept_label']} {lm['core_data']}"
        for lm in landmarks
    ]
    landmark_token_lists = [_tokenize(t) for t in landmark_texts]

    # Add query to corpus for IDF computation (ensures query terms get scored)
    query_tokens = _tokenize(query_text)
    full_corpus = landmark_token_lists + [query_tokens]

    idf = _compute_idf(full_corpus)

    query_vec = _build_tfidf_vector(query_tokens, idf)

    scored = []
    for lm, tokens in zip(landmarks, landmark_token_lists):
        lm_vec = _build_tfidf_vector(tokens, idf)
        similarity = _cosine_similarity(query_vec, lm_vec)

        # Relevancy multiplier — max 20% boost for a perfectly relevant memory
        relevancy_boost = 1.0 + (lm['relevancy_score'] * 0.2)
        final_score = similarity * relevancy_boost

        if final_score > 0.0:
            scored.append((final_score, lm))

    return sorted(scored, reverse=True)


# ---------------------------------------------------------------------------
# SONGLINE PATH BUILDER
# ---------------------------------------------------------------------------

def _build_narrative_path(landmark: dict,
                           max_edges: int = 5) -> dict:
    """
    For a matched Landmark, retrieves its strongest connected Songlines
    and builds the narrative path structure.

    Records a walk on each traversed Songline edge (increments walk_count).

    Returns a result dict ready for the caller.
    """
    songlines = get_songlines_for_landmark(landmark['id'])

    # Take the strongest edges up to max_edges
    top_edges = sorted(songlines, key=lambda s: s['strength_score'], reverse=True)[:max_edges]

    # Record traversal on each walked edge
    for edge in top_edges:
        walk_songline(edge['id'])

    # Build narrative fragments from edges
    narrative_fragments = []
    for edge in top_edges:
        direction = "→" if edge['origin_id'] == landmark['id'] else "←"
        narrative_fragments.append({
            'songline_id':    edge['id'],
            'direction':      direction,
            'connected_to':   edge['destination_id'] if direction == "→" else edge['origin_id'],
            'narrative':      edge['narrative_context'],
            'strength':       edge['strength_score'],
            'walk_count':     edge['walk_count'],
        })

    return {
        'landmark_id':    landmark['id'],
        'concept_label':  landmark['concept_label'],
        'core_data':      landmark['core_data'],
        'relevancy_score': landmark['relevancy_score'],
        'songlines':      narrative_fragments,
    }


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE
# ---------------------------------------------------------------------------

def query(text: str, top_n: int = 5,
          query_signature: str = None) -> list[dict]:
    """
    Main entry point. Query the SonglineMemory landscape.

    Takes a raw query string. Returns up to top_n results, each containing:
        - The matched Landmark (label, core_data, relevancy_score)
        - Its strongest connected Songlines (narrative path)
        - The similarity score that ranked it

    query_signature: optional unique string identifying this query context.
    Used to increment unique_query_count on recalled Landmarks.
    If None, defaults to the query text itself (reasonable for most uses).

    Side effects:
        - Increments recall_count on matched Landmarks
        - Increments unique_query_count when query_signature is new
        - Increments walk_count on traversed Songline edges

    Example:
        results = query("TF-IDF compression encoder")
        for r in results:
            print(r['concept_label'], r['score'])
            for s in r['songlines']:
                print("  →", s['narrative'])
    """
    if not text or not text.strip():
        return []

    # Use query text as signature if none provided
    sig = query_signature or text.strip()

    landmarks = _load_active_landmarks()
    if not landmarks:
        return []

    scored = _score_landmarks(text, landmarks)
    top = scored[:top_n]

    results = []
    for score, lm in top:
        # Record recall — increments counters, updates last_accessed
        updated_lm = recall_landmark(lm['id'], query_signature=sig)
        if updated_lm:
            lm.update(updated_lm)

        # Build narrative path from Songline edges
        path = _build_narrative_path(lm)
        path['score'] = round(score, 4)

        results.append(path)

    return results


def walk_path(landmark_id: int) -> dict | None:
    """
    Direct path walk from a known Landmark ID.
    Useful when the caller already knows which Landmark they want
    and just needs the full Songline narrative context around it.

    Returns the narrative path dict or None if not found.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, concept_label, core_data, relevancy_score FROM landmarks WHERE id = ?",
        (landmark_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    lm = dict(row)
    recall_landmark(landmark_id)
    return _build_narrative_path(lm)


def format_results(results: list[dict]) -> str:
    """
    Formats query results into a clean human/LLM readable string.
    Useful for injecting retrieved memory into an LLM prompt context.

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
        lines.append(f"RELEVANCY: {r['relevancy_score']:.2f}")

        if r['songlines']:
            lines.append("SONGLINES:")
            for s in r['songlines']:
                direction = s['direction']
                lines.append(
                    f"  {direction} \"{s['narrative']}\" "
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
        print("No results. Add some Landmarks first via Compressor + add_landmark().")
    else:
        print(format_results(results))

    print(f"{len(results)} result(s) returned.")
