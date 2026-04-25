🏜️ SonglineMemory
Graph-based AI memory that strengthens with use, not compute.
The first open-source AI memory system that gets smarter with use, runs on hardware you already own, and requires nothing you don't already have.
SonglineMemory is a locally hosted, graph-style AI memory architecture that delivers semantic relationship traversal, weighted edge scoring, and multi-tier memory consolidation — with zero vector database overhead, zero cloud dependency, and zero external Python dependencies.
It runs on an 8GB machine. It fits in a single `.db` file. It gets smarter the longer it runs.
---
The Problem With Every Other Memory System
If you've tried to give an LLM persistent memory, you've hit one of two walls:
Wall 1 — Vector databases require significant RAM, GPU acceleration, and infrastructure complexity to store and query floating-point embedding arrays. They work. They're also overkill for most use cases and impossible to run locally on repurposed hardware.
Wall 2 — Flat-file approaches (summarize everything, score what survives, write it back) treat memory as a cleanup problem. They miss the point. Memory isn't a list of facts — it's a web of relationships between facts. Pruning a flat file doesn't capture why two memories connect or how often that connection gets used.
SonglineMemory solves both problems with a different architectural premise entirely.
---
The Insight
In 2021, Monash University published research demonstrating that the ancient Australian Aboriginal Songline memory technique was measurably superior to the Greek "Memory Palace" for retention and recall. The key difference: Songlines store knowledge as narrative paths across a landscape, not as isolated locations in a room.
Large Language Models are trained on narrative. They predict narrative. They reason through narrative. Storing memory as narrative edges between compressed fact-nodes isn't just a design choice — it's architecturally aligned with how LLMs actually work.
This is the foundation of SonglineMemory.
---
How It Works
SonglineMemory has two primary components and three supporting systems.
Landmarks (Nodes)
Raw input — dictated, typed, or agent-generated — passes through a two-pass compressor that segments by topic, strips noise, scores sentences by information density, and extracts a compressed fact with a concept label. That compressed fact becomes a Landmark: a node in the memory graph stored in SQLite.
Songlines (Edges)
When a new Landmark is placed, SonglineMemory automatically scores it against all existing active Landmarks using TF-IDF cosine similarity and weaves Songline narrative edges to its nearest neighbors. Every node arrives already connected. The graph builds itself.
Songlines carry a `strength_score` that grows with use. Heavily traversed connections strengthen. Neglected ones decay. The graph reflects actual usage patterns, not just initial similarity.
The SongKeeper
The autonomous background consolidation cycle. Runs during idle time. Mimics biological sleep consolidation.
The SongKeeper classifies Landmarks, adjusts relevancy scores, and strengthens or weakens Songline edges based on traversal frequency. When a Landmark passes all three promotion gates, the SongKeeper graduates it from active memory into the Lore — carrying its full relationship history with it.
Every decision the SongKeeper makes is written to the Dream Diary (`DREAMS.md`) — a human-readable log you can review at any time. Promoted. Archived. Strengthened. Weakened. Every call, documented. No black boxes.
Promotion gates (all three must pass):
Minimum relevancy score
Minimum recall count
Minimum unique query count
Only memories that have proven themselves across multiple distinct contexts survive in active memory. Everything else graduates to long-term storage. Nothing is deleted — it changes address.
The Lore
The long-term knowledge layer. A persistent, accumulated knowledge base that grows richer over time and never forgets.
The Lore was conceptually inspired by Andrej Karpathy's LLM Wiki proposal — the idea that an AI system should maintain a living, compounding knowledge base rather than re-deriving facts from raw documents on every query. Where Karpathy's approach is a flat document synthesis layer, SonglineMemory's Lore is graph-aware. Memories don't just get written into the Lore — they earn their way in through the SongKeeper's three-gate promotion system, arriving with their full Songline relationship history intact.
Active SonglineMemory holds what's current and relevant. The Lore holds everything that was ever proven worth keeping. The two databases together — one live, one permanent — give you a complete memory architecture in a fraction of the footprint of any vector-based alternative.
The Dream Diary
`DREAMS.md` — generated automatically by the SongKeeper after every consolidation cycle. Human-readable. Append-only. Shows exactly what the system decided, what it kept, what it graduated, and why. Trust through transparency.
---
The Architecture
```
Raw Input (dictated / typed / agent)
    ↓
compressor.py       — segment by topic, strip noise, extract compressed facts
    ↓
core_memory.py      — place Landmark in SQLite, all DB operations
    ↓
weave.py            — auto-score neighbors, generate Songline edges
    ↓
retrieval.py        — TF-IDF query engine, Songline traversal, ranked results
    ↓
songkeeper.py       — async consolidation, edge strengthening, Lore graduation
    ↓
DREAMS.md           — human-readable log of every SongKeeper decision
```
Stack:
Language: Python 3.x
Database: SQLite (`landscape.db`)
Dependencies: zero. Pure Python stdlib + `math` module.
Target hardware: any machine with 4GB+ RAM
The entire system — active memory, long-term knowledge, relationship graph,
and full audit log — lives in a single `.db` file you can copy with one command.
---
Why SQLite Over a Vector DB
	Vector DB	SonglineMemory
RAM requirement	8–32GB+	4GB+
GPU required	Often	Never
External dependencies	Many	Zero
Relationship storage	Mathematical proximity	Narrative edges
Memory consolidation	Manual / flat-file	Graph-based, automatic
Portability	Complex	Single `.db` file
Runs locally	Difficult	Yes, by design
The graph relationships in SonglineMemory aren't approximated by vector distance — they're explicit, weighted, narrative connections that strengthen or decay based on actual usage. You get graph database behavior without graph database hardware requirements.
---
Compression
SonglineMemory's compressor uses TF-IDF weighted cosine similarity — the same scoring engine on both the write side (compression) and the read side (retrieval). This means the system scores semantic similarity consistently across the full pipeline. No hash functions. No semantic drift. Meaning distance is preserved from input to storage to recall.
Zero external dependencies. Pure Python `math` module.
---
Quick Start
```bash
# Clone the repo
git clone https://github.com/JoshinPgh/SonglineMemory.git
cd SonglineMemory

# Initialize the database and run the pipeline test
python3 pipeline.py
```
No pip installs. No environment setup. No Docker. No cloud accounts.
The first run initializes `landscape.db`, compresses the test input, places Landmarks, and weaves the first Songlines. You have a working memory graph in under 10 seconds.
---
Write Memory
```python
from pipeline import remember

remember("The encoder uses TF-IDF weighted cosine similarity to preserve "
         "meaning distance between landmarks.")
```
Query Memory
```python
from retrieval import query, format_results

results = query("TF-IDF encoder compression")
print(format_results(results))
```
---
Roadmap
Phase 1 — Core (complete)
[x] Two-pass context compressor
[x] Landmark placement
[x] Auto-Songline generation on write
[x] TF-IDF retrieval engine
[x] Songline traversal and narrative path builder
Phase 2 — SongKeeper (in progress)
[ ] Async consolidation cycle
[ ] Edge strength scoring and decay
[ ] Three-gate Lore promotion
[ ] Dream Diary (`DREAMS.md`) export
Phase 3 — Lore & Multi-model
[ ] Lore wiki layer with full-text search
[ ] Multi-model API layer (Claude, Gemini, Gemma shared memory)
[ ] Historical backfill pipeline
[ ] Chrome extension (read/write from browser)
---
Attribution
SonglineMemory is named in honor of the Australian Aboriginal Songline tradition — the oldest continuously practiced memory and knowledge transmission system in human history. Research published by Monash University (2021) demonstrated its measurable superiority over Western mnemonic techniques for retention and recall.
We build on their shoulders with respect and acknowledgment.
---
Built By
Josh Geldrich / JSG Labs — Geldrich Corp
Architected and built in collaboration with Claude (Anthropic).
"Memory can never be perfect, persistent, or eternal. Stuff has to get kicked off the back to make room for newer, developing relevancy. It's a bucket designed to be selectively leaky."
— Josh Geldrich, Geldrich Corp
