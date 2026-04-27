"""
compressor.py — SonglineMemory Context Compressor
JSG Labs / Geldrich Corp

Phase 1, Item 1.

Takes raw, verbose, dictated input and returns a list of
(concept_label, compressed_fact) tuples ready for add_landmark().

Zero external dependencies. Pure Python + re + string.

Two-pass design:
    Pass 1 — Segment by topic pivot detection
    Pass 2 — Per-segment: strip noise, score sentences, extract core fact + label

Design notes:
    - Dictated input clusters by topic before pivoting. We cut at the pivot.
    - Sentence-level segmentation is too granular.
    - Full-turn segmentation misses multi-fact blocks.
    - Intent is non-negotiable. We preserve meaning, not just keywords.
    - Output feeds directly into add_landmark(concept_label, core_data).
"""

import re
import string


# ---------------------------------------------------------------------------
# PIVOT SIGNALS — words/phrases that signal a topic shift in dictated speech
# ---------------------------------------------------------------------------

PIVOT_PHRASES = [
    # Hard pivots
    "anyway", "moving on", "next thing", "separate topic", "different topic",
    "on another note", "switching gears", "back to", "let me pivot",
    "different subject", "actually", "hold on", "wait",
    # Soft pivots — still worth cutting on when followed by a new clause
    "also", "another thing", "one more thing", "and another",
    "oh and", "by the way", "btw", "side note",
]

# Noise words to strip (filler, hedges, verbal restarts common in dictation)
NOISE_PATTERNS = [
    r"\bum+\b", r"\buh+\b", r"\blike\b(?!\s+\w+ing)",  # 'like' as filler, not verb
    r"\byou know\b", r"\bi mean\b", r"\bkind of\b", r"\bsort of\b",
    r"\bbasically\b", r"\bliterally\b", r"\bactually\b", r"\bjust\b",
    r"\bso\b(?=\s)", r"\bwell\b(?=\s)", r"\bright\b(?=\s*[,?])",
    r"\bokay so\b", r"\bokay\b(?=\s)", r"\balright\b",
    r"\bi guess\b", r"\bi think\b", r"\bprobably\b",  # hedges
    r"\byeah\b", r"\byep\b", r"\bnope\b",
    r'^[\s\-—]+',  # strip leading dash/em-dash artifacts from pivot splits
]

# Sentence-ending punctuation for splitting within a segment
SENT_END = re.compile(r'(?<=[.!?])\s+')


# ---------------------------------------------------------------------------
# PASS 1 — Segment raw input by topic pivot
# ---------------------------------------------------------------------------

def _segment_by_pivot(text: str) -> list[str]:
    """
    Splits raw input into topic-coherent blocks by detecting pivot phrases.
    Returns a list of raw text segments (may still be noisy).
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Build a pattern that matches pivot phrases at word boundaries,
    # preceded by sentence-ending punctuation or a comma+space.
    # We insert a split marker at each pivot.
    pivot_pattern = r'(?<=[.,!?])\s+(?=' + '|'.join(
        r'\b' + re.escape(p) + r'\b' for p in PIVOT_PHRASES
    ) + r')'

    raw_segments = re.split(pivot_pattern, text, flags=re.IGNORECASE)

    # Also split on hard sentence boundaries if a segment is very long
    # (> 400 chars = likely multi-topic blob that missed a pivot word)
    final_segments = []
    for seg in raw_segments:
        if len(seg) > 400:
            # Split on sentence boundaries and re-join into ~200-char chunks
            sentences = SENT_END.split(seg)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) < 250:
                    chunk = (chunk + " " + s).strip()
                else:
                    if chunk:
                        final_segments.append(chunk)
                    chunk = s
            if chunk:
                final_segments.append(chunk)
        else:
            final_segments.append(seg)

    cleaned = []
    for s in final_segments:
        s = s.strip()
        for p in PIVOT_PHRASES:
            s = re.sub(r'(?i)^' + re.escape(p) + r'[\s\-—]*', '', s).strip()
        s = re.sub(r'^[\s\-—]+', '', s).strip()
        if s:
            cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# PASS 2 — Per-segment noise stripping and fact extraction
# ---------------------------------------------------------------------------

def _strip_noise(text: str) -> str:
    """Removes filler words, hedges, and verbal restarts."""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Clean up leftover double spaces and leading/trailing punctuation artifacts
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'^[,\s]+', '', text)
    text = re.sub(r'[,\s]+$', '', text)
    # Strip isolated punctuation artifacts (e.g. trailing ", ?" or ", .")
    text = re.sub(r'\s*,\s*([.?!])$', r'\1', text)
    text = re.sub(r',\s*,', ',', text)  # double commas from noise removal
    return text.strip()


def _score_sentences(sentences: list[str]) -> list[tuple[float, str]]:
    """
    Scores each sentence by information density.
    Signal: longer sentences with more unique non-stop words score higher.
    Returns list of (score, sentence) tuples.
    """
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

    scored = []
    for sent in sentences:
        if not sent.strip():
            continue
        words = sent.lower().translate(
            str.maketrans('', '', string.punctuation)
        ).split()
        content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        unique_content = len(set(content_words))
        # Score = unique content word density * mild length bonus (caps at 20 words)
        length_bonus = min(len(words), 20) / 20
        score = unique_content * (1 + length_bonus * 0.3)
        scored.append((score, sent.strip()))

    return sorted(scored, reverse=True)


# REPLACEMENT for _extract_concept_label() in compressor.py
# Drop this in replacing the existing function — lines 154-178

def _extract_concept_label(text: str) -> str:
    """
    Derives a short concept label from the compressed fact.
    Strategy: take the first meaningful noun phrase — first 3-5 content words,
    title-cased, stripped of leading verbs of being AND pivot phrases.
    """
    SKIP_LEAD = {
        # Verbs of being
        'is', 'are', 'was', 'were', 'the', 'a', 'an', 'we', 'i',
        "i'm", "it's", "we're", "they're", "that's", "there's",
        'it', 'this', 'that', 'there', 'so', 'and', 'but', 'im',
        # Pivot phrases — single words that leak from segment boundaries
        'anyway', 'moving', 'next', 'separate', 'different', 'another',
        'also', 'additionally', 'furthermore', 'however', 'meanwhile',
        'switching', 'back', 'actually', 'hold', 'wait', 'oh', 'btw',
        'on', 'by', 'the', 'way', 'side', 'note', 'one', 'more',
    }

    words = text.split()
    content_words = []
    for w in words:
        clean = w.strip(string.punctuation).lower()
        if clean and clean not in SKIP_LEAD:
            content_words.append(w.strip(string.punctuation))
        if len(content_words) >= 5:
            break

    if not content_words:
        # Fallback: just take first 4 words
        content_words = words[:4]

    label = ' '.join(content_words[:4]).title()
    return label if label else "Unlabeled Concept"


def _compress_segment(raw_segment: str) -> tuple[str, str] | None:
    """
    Takes one raw topic segment and returns (concept_label, compressed_fact).
    Returns None if the segment has no extractable content.

    Two-pass per segment:
        1. Strip noise
        2. Score sentences, take highest-signal sentence as the core fact
    """
    cleaned = _strip_noise(raw_segment)
    if len(cleaned) < 10:
        return None

    # Split into sentences
    sentences = SENT_END.split(cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return None

    # Score and pick the top sentence as core fact
    scored = _score_sentences(sentences)

    if not scored:
        return None

    # Core fact = top-scoring sentence
    core_fact = scored[0][1]
    # Strip trailing orphaned punctuation artifacts (e.g. ", ?" from noise removal)
    core_fact = re.sub(r'\s*,\s*\?$', '.', core_fact)
    core_fact = re.sub(r'\s*,\s*\.$', '.', core_fact)

    # If there are additional high-scoring sentences (score within 30% of top),
    # append them — this preserves compound facts in dense segments.
    top_score = scored[0][0]
    supplements = [
        s for score, s in scored[1:]
        if score >= top_score * 0.70 and s != core_fact
    ]

    if supplements:
        core_fact = core_fact.rstrip('.') + '. ' + ' '.join(supplements[:2])

    concept_label = _extract_concept_label(core_fact)

    return (concept_label, core_fact)


# ---------------------------------------------------------------------------
# PUBLIC INTERFACE
# ---------------------------------------------------------------------------

def compress(raw_input: str) -> list[tuple[str, str]]:
    """
    Main entry point. Takes raw dictated input (one turn or a full block).
    Returns a list of (concept_label, compressed_fact) tuples.

    Each tuple maps directly to add_landmark(concept_label, compressed_fact).

    Returns an empty list if nothing extractable was found.

    Example:
        results = compress("So I'm building this memory system, right?
                            It uses SQLite and pure Python.
                            Anyway, the other thing is the encoder needs
                            to use TF-IDF to preserve meaning distance.")

        # Returns something like:
        # [
        #   ("Memory System SQLite", "Building memory system uses SQLite pure Python."),
        #   ("Encoder Tfidf Meaning", "Encoder needs TF-IDF to preserve meaning distance.")
        # ]
    """
    if not raw_input or not raw_input.strip():
        return []

    segments = _segment_by_pivot(raw_input)
    results = []

    for seg in segments:
        result = _compress_segment(seg)
        if result is not None:
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# QUICK TEST — run this file directly to verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_INPUT = """
    So um, I'm building this memory system, right, it's basically a graph-based
    thing that uses SQLite and pure Python — no external dependencies at all,
    that's non-negotiable for version one. The whole point is it runs local,
    no cloud, no cost. Anyway, moving on — the other thing is the encoder.
    The encoder needs to use TF-IDF weighted cosine similarity to preserve
    meaning distance between landmarks, because the old hash compressor was
    just destroying semantic meaning entirely, you know? Like it preserved
    identity but not distance, which means you can't actually walk toward
    a memory you can't measure distance to. Also — the Walkabout cycle,
    that's the dream consolidation process. It strengthens or weakens
    narrative edges based on how often they get traversed. Closer to
    biological synaptic consolidation than anything else in this space.
    """

    print("=== COMPRESSOR TEST ===\n")
    print(f"INPUT ({len(TEST_INPUT)} chars):\n{TEST_INPUT.strip()}\n")
    print("--- OUTPUT ---")

    facts = compress(TEST_INPUT)

    if not facts:
        print("No facts extracted. Check input.")
    else:
        for i, (label, fact) in enumerate(facts, 1):
            print(f"\n[{i}] LABEL: {label}")
            print(f"    FACT:  {fact}")

    print(f"\n{len(facts)} landmark(s) ready for placement.")
