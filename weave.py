"""
weave.py — SonglineMemory Auto-Songline Weaver
JSG Labs / Geldrich Corp

Phase 1, Item 3.

When a new Landmark is placed, this module automatically finds its
nearest neighbors in the active landscape and weaves a Songline
narrative edge connecting them.

This is what makes SonglineMemory a graph instead of a flat list.
Every new node arrives already connected.

Pipeline (called automatically from pipeline.py after add_landmark()):
    new Landmark placed
        → score against all active Landmarks via TF-IDF cosine
        → take top N neighbors above similarity threshold
        → generate template-based narrative for each edge
        → call add_songline() to weave each connection
        → return list of woven songline IDs

v1 Design note:
    Songline narratives are template-based. No LLM dependency.
    Templates use concept labels and core data of both endpoints
    to produce coherent, useful connective tissue.
    LLM-generated narratives arrive in v3 with the multi-model
    API layer on 851Office-1.

Zero external dependencies. Pure Python + math.
"""

import math
import string
import sqlite3
import random

from core_memory import (
    DB_NAME,
    add_songline,
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# How many neighbors to connect a new Landmark to on arrival
MAX_NEIGHBORS = 3

# Minimum similarity score to bother weaving a Songline
# Below this = landmarks are too unrelated to connect meaningfully
SIMILARITY_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# STOP WORDS (local copy — avoids circular import with retrieval.py)
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
# TF-IDF (local — same math as retrieval.py, kept separate to avoid
# circular imports. If this grows, extract to a shared tfidf.py util.)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [
        w for w in
        text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        if w not in STOP_WORDS and len(w) > 2
    ]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def _compute_idf(corpus: list[list[str]]) -> dict[str, float]:
    N = len(corpus)
    df = {}
    for doc_tokens in corpus:
        for term in set(doc_tokens):
            df[term] = df.get(term, 0) + 1
    return {
        term: math.log((1 + N) / (1 + count)) + 1
        for term, count in df.items()
    }


def _build_tfidf_vector(tokens: list[str],
                         idf: dict[str, float]) -> dict[str, float]:
    tf = _term_freq(tokens)
    return {t: tf_val * idf.get(t, 0.0) for t, tf_val in tf.items()}


def _cosine_similarity(vec_a: dict[str, float],
                        vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot    = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vec_b)
    mag_a  = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b  = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# NEIGHBOR SCORING
# ---------------------------------------------------------------------------

def _load_existing_landmarks(exclude_id: int) -> list[dict]:
    """
    Loads all active Landmarks except the one just placed.
    We don't connect a node to itself.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, concept_label, core_data
        FROM landmarks
        WHERE memory_tier = 'active' AND id != ?
    """, (exclude_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def _find_neighbors(new_landmark: dict,
                    existing: list[dict]) -> list[tuple[float, dict]]:
    """
    Scores existing Landmarks against the new one.
    Returns list of (similarity_score, landmark) above threshold,
    sorted descending, capped at MAX_NEIGHBORS.
    """
    if not existing:
        return []

    new_text    = f"{new_landmark['concept_label']} {new_landmark['core_data']}"
    new_tokens  = _tokenize(new_text)

    existing_texts  = [f"{lm['concept_label']} {lm['core_data']}" for lm in existing]
    existing_tokens = [_tokenize(t) for t in existing_texts]

    corpus = existing_tokens + [new_tokens]
    idf    = _compute_idf(corpus)

    new_vec = _build_tfidf_vector(new_tokens, idf)

    scored = []
    for lm, tokens in zip(existing, existing_tokens):
        vec        = _build_tfidf_vector(tokens, idf)
        similarity = _cosine_similarity(new_vec, vec)
        if similarity >= SIMILARITY_THRESHOLD:
            scored.append((similarity, lm))

    return sorted(scored, key=lambda x: x[0], reverse=True)[:MAX_NEIGHBORS]


# ---------------------------------------------------------------------------
# TEMPLATE-BASED NARRATIVE GENERATION
# v1 — no LLM dependency. Templates produce coherent connective tissue
# using the concept labels of both Landmark endpoints.
# v3 will replace this with LLM-generated narratives via 851Office-1.
# ---------------------------------------------------------------------------

# Template sets by relationship type
# Selected based on similarity score band — stronger similarity
# gets a tighter, more specific template.

TEMPLATES_STRONG = [
    "The knowledge of {origin} directly informs {destination}, sharing core conceptual territory.",
    "{origin} and {destination} are closely related — understanding one deepens the other.",
    "A strong thread runs between {origin} and {destination}, each reinforcing the other's meaning.",
    "{destination} emerges as a natural extension of the principles established in {origin}.",
    "The path from {origin} leads directly to {destination} — these landmarks anchor the same region.",
]

TEMPLATES_MODERATE = [
    "{origin} and {destination} share common ground — a connecting thread worth following.",
    "From {origin}, a Songline runs toward {destination}, linking related but distinct ideas.",
    "{destination} was reached while walking the territory of {origin}.",
    "The landscape between {origin} and {destination} holds connected meaning.",
    "{origin} casts a shadow that reaches {destination} — related context, different ground.",
]

TEMPLATES_WEAK = [
    "{origin} and {destination} occupy adjacent territory in the landscape.",
    "A faint path connects {origin} to {destination} — tangentially related.",
    "Walking from {origin}, {destination} appears on the horizon — loosely connected.",
    "{destination} was encountered in the same region as {origin}.",
    "A minor Songline links {origin} and {destination} — worth noting, not yet proven.",
]


def _generate_narrative(origin: dict, destination: dict,
                         similarity: float) -> str:
    """
    Selects a template based on similarity score band and
    fills it with the concept labels of both endpoints.

    Strong  : similarity >= 0.50
    Moderate: similarity >= 0.25
    Weak    : similarity >= SIMILARITY_THRESHOLD
    """
    origin_label      = origin['concept_label'].title()
    destination_label = destination['concept_label'].title()

    if similarity >= 0.50:
        template = random.choice(TEMPLATES_STRONG)
    elif similarity >= 0.25:
        template = random.choice(TEMPLATES_MODERATE)
    else:
        template = random.choice(TEMPLATES_WEAK)

    return template.format(
        origin=origin_label,
        destination=destination_label
    )


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE
# ---------------------------------------------------------------------------

def weave(new_landmark_id: int,
          new_concept_label: str,
          new_core_data: str) -> list[int]:
    """
    Main entry point. Called immediately after add_landmark().

    Finds nearest neighbors in the active landscape and weaves
    a Songline edge to each one.

    Returns list of newly created Songline IDs.
    (Empty list if no neighbors found above threshold — valid for
    the very first Landmark placed, or highly unique facts.)

    Example (from pipeline.py):
        landmark_id = add_landmark(concept, data)
        songline_ids = weave(landmark_id, concept, data)
    """
    new_landmark = {
        'id':            new_landmark_id,
        'concept_label': new_concept_label,
        'core_data':     new_core_data,
    }

    existing   = _load_existing_landmarks(exclude_id=new_landmark_id)
    neighbors  = _find_neighbors(new_landmark, existing)

    if not neighbors:
        print(f"WEAVE: No neighbors found for '{new_concept_label}' "
              f"above threshold {SIMILARITY_THRESHOLD}. "
              f"Landmark stands alone for now.")
        return []

    woven_ids = []
    for similarity, neighbor in neighbors:
        narrative = _generate_narrative(
            origin=new_landmark,
            destination=neighbor,
            similarity=similarity
        )

        songline_id = add_songline(
            origin_id=new_landmark_id,
            destination_id=neighbor['id'],
            narrative=narrative
        )

        woven_ids.append(songline_id)
        print(f"WEAVE: '{new_concept_label}' → '{neighbor['concept_label']}' "
              f"(similarity: {similarity:.3f} | songline: {songline_id})")

    return woven_ids


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== WEAVE TEST ===")
    print("Weave runs after add_landmark() via pipeline.py.")
    print("Run pipeline.py to test the full write pipeline end to end.")
    print()
    print(f"Config: MAX_NEIGHBORS={MAX_NEIGHBORS} | "
          f"SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}")
