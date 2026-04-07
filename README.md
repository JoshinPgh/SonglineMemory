🏜️ SonglineMemory: The Aboriginal Narrative Landscape for AI
A locally hosted, graph-style AI memory system based on the oldest continuous memory technique in human history, supercharged by extreme vector compression.

Current AI memory solutions rely heavily on either resource-intensive vector databases or rigid spatial geometries (inspired by the ancient Greek "Memory Palace"). While effective, these systems struggle with organic semantic relationships, context decay, and massive memory overhead.

SonglineMemory takes a different approach. Inspired by the 2021 Monash University study which proved the ancient Australian Aboriginal memory tool to be vastly superior for retention, this system maps AI context to a continuous conceptual landscape connected by narrative.

🧠 The Philosophy: Landmarks & Songlines
Instead of dropping data into a virtual "room" or relying on raw mathematical proximity, SonglineMemory uses two core concepts:

Landmarks (Nodes): Core facts, compressed entities, and user intents mapped to a virtual topological graph.

Songlines (Edges): Brief, AI-generated narratives that logically and contextually connect one Landmark to another.

Large Language Models are fundamentally designed to predict and understand narrative flow. By storing memory relationships as brief stories rather than isolated data points, the AI can "walk the path" of its own memory, retrieving not just the target fact, but the nuanced history of how that fact was reached.

⚡ Extreme Compression via TurboQuant
To make this system incredibly lightweight, SonglineMemory's data pipeline utilizes the mathematical principles of Google's TurboQuant algorithm to compress semantic data before it ever hits the database. This software-only breakthrough reduces memory footprint by up to 6x and indexing time to near-zero without losing accuracy.

PolarQuant: Instead of creating a metadata tax, the system applies a random orthogonal rotation to standard Cartesian inputs, mapping them cleanly to polar coordinates (radius and angle). This simplifies the data geometry, allowing for aggressive quantization that shrinks the memory footprint drastically.

QJL (Quantized Johnson-Lindenstrauss): To prevent "hallucinations" or semantic drift from the compression, a 1-bit QJL transform acts as a mathematical error-checker. It eliminates residual bias, ensuring the compressed data remains perfectly aligned with the original thought.

🛠️ The Architecture
This project was built with a strict Zero-Outside-Funding / Bootstrapper's Mindset. It is designed to run efficiently on local machines without eating up system RAM or relying on expensive cloud infrastructure.

Language: Python

Database: Local SQLite (landscape.db)

Dependencies: Minimal by design. Runs entirely natively via Python's built-in libraries.

⚙️ How It Works (The Core Loop)
Parse & Compress: Extracts the core fact and passes it through the TurboQuant (PolarQuant/QJL) compression pipeline.

Place the Landmark: Saves the highly compressed data point into the SQLite landmarks table.

Weave the Songline: Queries recent Landmarks and generates a connecting narrative, saving the pathway into the songlines table.

🚀 Why SQLite over Vector/Cloud DBs?
Ephemeral safety: No accidental data wiping from Docker container restarts.

Low overhead: Uses system resources only during the exact millisecond a query runs.

Portability: The entire AI memory footprint exists in a single .db file.

Architected and built in collaboration with Gemini 3.1 Pro.