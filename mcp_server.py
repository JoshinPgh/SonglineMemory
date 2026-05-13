"""
mcp_server.py — SonglineMemory MCP Server
JSG Labs / Geldrich Corp

Wraps the SonglineMemory REST API (api.py / Flask, port 5000)
in MCP protocol so Claude.ai web app can connect via
Settings → Connectors → Remote MCP (HTTP/SSE).

Runs on port 5001 alongside the Flask API.
Proxies all tool calls to the local Flask server.

Tools exposed:
    remember        — write new memory into the landscape
    query_memory    — query the active landscape
    search_lore     — search the Lore knowledge base
    health_check    — confirm API is live

To run:
    python3 /mnt/SonglineMemory/mcp_server.py

To run as background service, add to systemd (see songline.service pattern).

Dependencies:
    pip3 install mcp --break-system-packages
    (installs mcp, uvicorn, httpx, starlette automatically)
"""

import os
import sys
import httpx

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

sys.path.insert(0, '/mnt/SonglineMemory')

ENV_PATH    = '/mnt/SonglineMemory/.env'
FLASK_BASE  = 'http://127.0.0.1:5000'   # local Flask API
MCP_PORT    = 5001                        # MCP server port


# ---------------------------------------------------------------------------
# LOAD API KEY
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    if not os.path.exists(ENV_PATH):
        print(f"[mcp] ERROR: .env not found at {ENV_PATH}")
        sys.exit(1)
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('SONGLINE_API_KEY='):
                key = line.split('=', 1)[1].strip()
                if key:
                    return key
    print("[mcp] ERROR: SONGLINE_API_KEY not found in .env")
    sys.exit(1)


API_KEY = _load_api_key()
HEADERS = {
    'X-API-Key':    API_KEY,
    'Content-Type': 'application/json',
}


# ---------------------------------------------------------------------------
# MCP SERVER
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="SonglineMemory",
    instructions=(
        "SonglineMemory is a locally-hosted, graph-based AI memory system "
        "for Geldrich Corp / JSG Labs. Use it to store and retrieve memories, "
        "facts, and knowledge across conversations. "
        "Call remember() to write new information. "
        "Call query_memory() to find relevant memories. "
        "Call search_lore() to search the long-term knowledge base."
    ),
)


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
def health_check() -> str:
    """
    Check that the SonglineMemory API is live and reachable.
    Returns status and version info.
    """
    with httpx.Client(timeout=10) as client:
        r = client.get(f"{FLASK_BASE}/health", headers=HEADERS)
        r.raise_for_status()
        data = r.json()
    return f"SonglineMemory API is {data['status']} — version {data.get('version', 'unknown')}"


@mcp.tool()
def remember(text: str, source: str = None) -> str:
    """
    Write new memory into SonglineMemory.

    Stores the text as one or more Landmarks in the active memory graph,
    automatically connecting them to related existing memories via Songline edges.

    Args:
        text:   Raw text to remember. Can be dictated, typed, or agent-generated.
                The system will compress and extract key facts automatically.
        source: Optional tag identifying where this memory came from
                (e.g. 'gemini_wiki', 'claude_session', 'manual').

    Returns:
        Summary of what was stored — landmark count, songlines woven.
    """
    payload = {'text': text}
    if source:
        payload['source'] = source

    with httpx.Client(timeout=30) as client:
        r = client.post(f"{FLASK_BASE}/remember", headers=HEADERS, json=payload)
        r.raise_for_status()
        data = r.json()

    placed   = data.get('landmarks_placed', 0)
    woven    = data.get('songlines_woven', 0)
    results  = data.get('results', [])

    lines = [f"Stored {placed} landmark(s), wove {woven} songline(s)."]
    for r in results:
        lines.append(f"  • {r['concept_label']}: {r['core_data'][:80]}")

    return '\n'.join(lines)


@mcp.tool()
def query_memory(text: str, top_n: int = 5) -> str:
    """
    Query the active SonglineMemory landscape for relevant memories.

    Searches landmarks using TF-IDF cosine similarity, returns the most
    relevant memories along with their narrative Songline connections.

    Args:
        text:   What you want to recall. Natural language query.
        top_n:  Maximum number of results to return (default 5).

    Returns:
        Formatted memory results ready for use as context.
        Each result includes the landmark fact, relevancy score,
        and connected Songline narratives.
    """
    with httpx.Client(timeout=15) as client:
        r = client.get(
            f"{FLASK_BASE}/query",
            headers=HEADERS,
            params={'q': text, 'top_n': top_n, 'format': 'text'},
        )
        r.raise_for_status()

    return r.text


@mcp.tool()
def search_lore(text: str = None, tag: str = None, limit: int = 20) -> str:
    """
    Search the SonglineMemory Lore — the long-term knowledge base.

    Lore contains memories that have proven themselves over time:
    high relevancy, high recall count, queried across multiple contexts.
    Think of it as the institutional knowledge layer — stable, trusted facts.

    Args:
        text:   Optional search text to filter results.
        tag:    Optional source tag filter (e.g. 'gemini_wiki').
        limit:  Maximum results to return (default 20).

    Returns:
        Formatted Lore entries matching the query.
    """
    params = {'limit': limit, 'format': 'text'}
    if text:
        params['q'] = text
    if tag:
        params['tag'] = tag

    with httpx.Client(timeout=15) as client:
        r = client.get(f"{FLASK_BASE}/lore", headers=HEADERS, params=params)
        r.raise_for_status()

    return r.text


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"\n[SonglineMemory MCP] Starting on port {MCP_PORT}")
    print(f"[SonglineMemory MCP] Proxying to Flask at {FLASK_BASE}")
    print(f"[SonglineMemory MCP] Tools: health_check | remember | query_memory | search_lore\n")

    mcp.run(transport='sse', port=MCP_PORT)
