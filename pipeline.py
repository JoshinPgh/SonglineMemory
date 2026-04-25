"""
pipeline.py — SonglineMemory Write Pipeline
JSG Labs / Geldrich Corp

Phase 1 — Complete write pipeline.

Single entry point for all new memory input.
Wires the full chain:

    raw input (dictated or text)
        → compressor.compress()         — segment + denoise + extract facts
        → core_memory.add_landmark()    — place each fact as a Landmark
        → weave.weave()                 — auto-connect to nearest neighbors
        → return summary

This is the only file an LLM agent or external caller needs to touch
to write new memory into SonglineMemory.

Usage:
    from pipeline import remember
    remember("So I'm building this memory system with SQLite and pure Python...")

Zero external dependencies. Pure Python stdlib.
"""

from compressor import compress
from core_memory import add_landmark, init_db
from weave import weave


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE
# ---------------------------------------------------------------------------

def remember(raw_input: str, verbose: bool = True) -> list[dict]:
    """
    Main entry point for all memory writes.

    Takes raw text (dictated, typed, or agent-generated).
    Returns a list of result dicts — one per Landmark placed.

    Each result dict contains:
        landmark_id   : int
        concept_label : str
        core_data     : str
        songline_ids  : list[int]  — IDs of Songlines woven on arrival

    Returns empty list if nothing extractable was found in the input.

    Example:
        from pipeline import remember

        results = remember(
            "The SongKeeper runs asynchronously during idle time. "
            "It promotes Landmarks that pass all three gate thresholds "
            "into the Lore for long-term storage."
        )

        for r in results:
            print(r['concept_label'], '— connected by', len(r['songline_ids']), 'songlines')
    """
    if not raw_input or not raw_input.strip():
        if verbose:
            print("[pipeline] Empty input. Nothing to remember.")
        return []

    if verbose:
        print(f"\n[pipeline] Input received ({len(raw_input)} chars). Compressing...\n")

    # Pass 1 — Compress raw input into (concept_label, core_data) tuples
    facts = compress(raw_input)

    if not facts:
        if verbose:
            print("[pipeline] compressor returned no extractable facts.")
        return []

    if verbose:
        print(f"[pipeline] {len(facts)} fact(s) extracted. Placing Landmarks...\n")

    results = []

    for concept_label, core_data in facts:
        # Pass 2 — Place each fact as a Landmark
        landmark_id = add_landmark(concept_label, core_data)

        # Pass 3 — Auto-weave Songlines to nearest neighbors
        songline_ids = weave(
            new_landmark_id=landmark_id,
            new_concept_label=concept_label,
            new_core_data=core_data,
        )

        results.append({
            'landmark_id':   landmark_id,
            'concept_label': concept_label,
            'core_data':     core_data,
            'songline_ids':  songline_ids,
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
    # Boot the DB if it doesn't exist yet
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
