"""
pipeline.py — SonglineMemory Write Pipeline
JSG Labs / Geldrich Corp

Phase 1 & 2 — Complete write pipeline.

Single entry point for all new memory input.
Wires the full chain:

    raw input (dictated, typed, or batch import)
        → compressor.compress()         — segment + denoise + extract facts
        → core_memory.add_landmark()    — place each fact as a Landmark
        → weave.weave()                 — auto-connect to nearest neighbors
        → return summary

This is the only file an LLM agent or external caller needs to touch
to write new memory into SonglineMemory.

Source tagging:
    Pass source="gemini_wiki" (or any string) to tag where a batch of
    memories came from. Stored in concept_label prefix for Lore queryability.
    Defaults to None (live session input, no tag needed).

Usage:
    from pipeline import remember

    # Live session input
    remember("So I'm building this memory system with SQLite and pure Python...")

    # Batch import with source tag
    remember(wiki_text, source="gemini_wiki")

Zero external dependencies. Pure Python stdlib.
"""

import sqlite3
import string

from compressor import compress
from core_memory import add_landmark, init_db, DB_NAME
from weave import weave


# ---------------------------------------------------------------------------
# DUPLICATE DETECTION
# ---------------------------------------------------------------------------

# How similar two concept labels need to be to count as a duplicate (0.0–1.0)
DUPLICATE_THRESHOLD = 0.80

def _tokenize_label(text: str) -> set[str]:
    """Simple token set for label comparison — lowercase, no punctuation."""
    stop = {'the','a','an','is','are','was','to','of','in','for','on','with'}
    return {
        w for w in
        text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        if w not in stop and len(w) > 2
    }

def _is_duplicate(concept_label: str,
                  existing_labels: list[str]) -> bool:
    """
    Returns True if concept_label is too similar to any existing active label.
    Uses Jaccard similarity on token sets — fast, zero dependencies.
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    tokens_new = _tokenize_label(concept_label)
    if not tokens_new:
        return False

    for existing in existing_labels:
        tokens_ex = _tokenize_label(existing)
        if not tokens_ex:
            continue
        intersection = len(tokens_new & tokens_ex)
        union        = len(tokens_new | tokens_ex)
        if union == 0:
            continue
        if intersection / union >= DUPLICATE_THRESHOLD:
            return True

    return False


def _load_active_labels() -> list[str]:
    """Loads all active concept labels from the database for duplicate checking."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT concept_label FROM landmarks WHERE memory_tier = 'active'"
    )
    labels = [row[0] for row in cursor.fetchall()]
    conn.close()
    return labels


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE
# ---------------------------------------------------------------------------

def remember(raw_input: str,
             verbose: bool = True,
             source: str = None) -> list[dict]:
    """
    Main entry point for all memory writes.

    Takes raw text (dictated, typed, or agent-generated).
    Returns a list of result dicts — one per Landmark placed.

    Each result dict contains:
        landmark_id   : int
        concept_label : str
        core_data     : str
        songline_ids  : list[int]  — IDs of Songlines woven on arrival
        source        : str | None — origin tag for batch imports

    source:
        Optional string identifying the origin of a batch import.
        e.g. "gemini_wiki", "obsidian_vault", "manual_backfill"
        When set, prepended to concept_label as "[source] label"
        so Lore entries are queryable by origin.
        Leave None for all live session input.

    Returns empty list if nothing extractable was found in the input.
    """
    if not raw_input or not raw_input.strip():
        if verbose:
            print("[pipeline] Empty input. Nothing to remember.")
        return []

    if verbose:
        src_tag = f" | source: {source}" if source else ""
        print(f"\n[pipeline] Input received ({len(raw_input)} chars{src_tag}). Compressing...\n")

    # Pass 1 — Compress raw input into (concept_label, core_data) tuples
    facts = compress(raw_input)

    if not facts:
        if verbose:
            print("[pipeline] compressor returned no extractable facts.")
        return []

    if verbose:
        print(f"[pipeline] {len(facts)} fact(s) extracted. Placing Landmarks...\n")

    results = []

    # Load existing labels once for duplicate checking across this batch
    existing_labels = _load_active_labels()

    for concept_label, core_data in facts:

        # Apply source tag to concept_label if provided
        tagged_label = f"[{source}] {concept_label}" if source else concept_label

        # Duplicate check — skip if too similar to an existing active landmark
        if _is_duplicate(tagged_label, existing_labels):
            if verbose:
                print(f"[pipeline] SKIPPED duplicate: '{tagged_label}'")
            continue

        # Pass 2 — Place each fact as a Landmark
        landmark_id = add_landmark(tagged_label, core_data)

        # Track in-session so we catch dupes within the same batch
        existing_labels.append(tagged_label)

        # Pass 3 — Auto-weave Songlines to nearest neighbors
        songline_ids = weave(
            new_landmark_id=landmark_id,
            new_concept_label=tagged_label,
            new_core_data=core_data,
        )

        results.append({
            'landmark_id':   landmark_id,
            'concept_label': tagged_label,
            'core_data':     core_data,
            'songline_ids':  songline_ids,
            'source':        source,
        })

        if verbose:
            print()

    if verbose:
        print(f"[pipeline] Complete. {len(results)} landmark(s) placed, "
              f"{sum(len(r['songline_ids']) for r in results)} songline(s) woven.\n")

    return results


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()

    TEST_INPUT = """
    So I'm building this memory system right, it uses SQLite and pure Python,
    zero external dependencies, that's non-negotiable.
    The whole thing runs locally, no cloud involved.
    Moving on — the encoder needs to use TF-IDF weighted cosine similarity
    to preserve meaning distance between landmarks, because the old hash
    compressor was destroying semantic meaning entirely.
    Also — the SongKeeper is the async background consolidation cycle.
    It mimics biological sleep. It strengthens or weakens Songline edges
    based on how often they get walked, and promotes Landmarks that pass
    all three gate thresholds into the Lore for long-term storage.
    """

    print("=== PIPELINE TEST ===\n")

    results = remember(TEST_INPUT)

    print("--- RESULTS ---")
    for r in results:
        print(f"\nLandmark [{r['landmark_id']}]: {r['concept_label']}")
        print(f"  Fact: {r['core_data'][:80]}{'...' if len(r['core_data']) > 80 else ''}")
        print(f"  Songlines woven: {len(r['songline_ids'])} {r['songline_ids']}")
