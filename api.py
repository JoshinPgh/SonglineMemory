"""
api.py — SonglineMemory API Layer
JSG Labs / Geldrich Corp

Phase 3 — The bridge.

Exposes SonglineMemory to any external caller over HTTP.
Runs as a lightweight Flask server on 851Office-1.
Accessible locally and via Tailscale from anywhere on the mesh network.

Endpoints:
    POST /remember          — write new memory into the landscape
    GET  /query             — query the active landscape
    GET  /lore              — search the Lore knowledge base
    GET  /health            — server health check

Authentication:
    Every request must include the API key in the header:
        X-API-Key: <your key>
    Key is loaded from /mnt/SonglineMemory/.env — never hardcoded,
    never in GitHub.

Usage examples (from any machine on Tailscale):
    # Health check
    curl http://192.168.1.41:5000/health -H "X-API-Key: YOUR_KEY"

    # Write memory
    curl -X POST http://192.168.1.41:5000/remember \
         -H "X-API-Key: YOUR_KEY" \
         -H "Content-Type: application/json" \
         -d '{"text": "SonglineMemory API layer is now live on 851Office-1."}'

    # Query memory
    curl "http://192.168.1.41:5000/query?q=Flask+API+layer" \
         -H "X-API-Key: YOUR_KEY"

    # Search Lore
    curl "http://192.168.1.41:5000/lore?q=memory+architecture" \
         -H "X-API-Key: YOUR_KEY"

To run:
    python3 /mnt/SonglineMemory/api.py

To run as background service (keep running after SSH disconnect):
    nohup python3 /mnt/SonglineMemory/api.py > /mnt/SonglineMemory/api.log 2>&1 &

To auto-start on boot, add to crontab:
    @reboot python3 /mnt/SonglineMemory/api.py > /mnt/SonglineMemory/api.log 2>&1 &

Dependencies:
    Flask — install once on 851Office-1:
        pip3 install flask --break-system-packages
"""

import os
import sys
from functools import wraps

# ---------------------------------------------------------------------------
# PATH SETUP — ensures imports find SonglineMemory modules
# ---------------------------------------------------------------------------

sys.path.insert(0, '/mnt/SonglineMemory')

from flask import Flask, request, jsonify
from pipeline import remember
from retrieval import query, search_lore, format_results, format_lore_results

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ENV_PATH = '/mnt/SonglineMemory/.env'
HOST     = '0.0.0.0'   # listen on all interfaces — local + Tailscale
PORT     = 5000


# ---------------------------------------------------------------------------
# LOAD API KEY FROM .env
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """
    Reads SONGLINE_API_KEY from .env file.
    Exits immediately if not found — no key, no server.
    """
    if not os.path.exists(ENV_PATH):
        print(f"[api] ERROR: .env file not found at {ENV_PATH}")
        print("[api] Run: echo 'SONGLINE_API_KEY=your_key' > /mnt/SonglineMemory/.env")
        sys.exit(1)

    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('SONGLINE_API_KEY='):
                key = line.split('=', 1)[1].strip()
                if key:
                    return key

    print("[api] ERROR: SONGLINE_API_KEY not found in .env")
    sys.exit(1)


API_KEY = _load_api_key()
app     = Flask(__name__)


# ---------------------------------------------------------------------------
# AUTH DECORATOR
# ---------------------------------------------------------------------------

def require_api_key(f):
    """
    Decorator that checks X-API-Key header on every protected endpoint.
    Returns 401 if missing or wrong.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if not key or key != API_KEY:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Valid X-API-Key header required.'
            }), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
@require_api_key
def health():
    """
    Health check. Confirms the API is running and key is valid.

    Response:
        { "status": "ok", "service": "SonglineMemory API" }
    """
    return jsonify({
        'status':  'ok',
        'service': 'SonglineMemory API',
        'version': '1.0',
    })


@app.route('/remember', methods=['POST'])
@require_api_key
def remember_endpoint():
    """
    Write new memory into SonglineMemory.

    Body (JSON):
        {
            "text":   "raw input text to remember",
            "source": "optional source tag e.g. gemini_wiki"  (optional)
        }

    Response:
        {
            "landmarks_placed": 3,
            "songlines_woven":  2,
            "results": [ { landmark_id, concept_label, core_data, songline_ids } ]
        }
    """
    data = request.get_json(silent=True)

    if not data or 'text' not in data:
        return jsonify({
            'error':   'Bad Request',
            'message': 'JSON body with "text" field required.'
        }), 400

    text   = data.get('text', '').strip()
    source = data.get('source', None)

    if not text:
        return jsonify({
            'error':   'Bad Request',
            'message': '"text" field cannot be empty.'
        }), 400

    results = remember(text, verbose=False, source=source)

    return jsonify({
        'landmarks_placed': len(results),
        'songlines_woven':  sum(len(r['songline_ids']) for r in results),
        'results':          results,
    })


@app.route('/query', methods=['GET'])
@require_api_key
def query_endpoint():
    """
    Query the active SonglineMemory landscape.

    Params:
        q      : query string (required)
        top_n  : max results to return (optional, default 5)
        format : 'json' or 'text' (optional, default 'json')
                 'text' returns LLM-injectable formatted string

    Response (json):
        {
            "query": "your query",
            "results": [ { landmark_id, concept_label, core_data,
                           relevancy_score, recall_count, score, songlines } ]
        }

    Response (text):
        Plain formatted string ready for LLM prompt injection.
    """
    q      = request.args.get('q', '').strip()
    top_n  = int(request.args.get('top_n', 5))
    fmt    = request.args.get('format', 'json')

    if not q:
        return jsonify({
            'error':   'Bad Request',
            'message': 'Query parameter "q" required.'
        }), 400

    results = query(q, top_n=top_n)

    if fmt == 'text':
        return format_results(results), 200, {'Content-Type': 'text/plain'}

    return jsonify({
        'query':   q,
        'count':   len(results),
        'results': results,
    })


@app.route('/lore', methods=['GET'])
@require_api_key
def lore_endpoint():
    """
    Search the Lore knowledge base directly.

    Params:
        q      : search text (optional)
        tag    : filter by source tag (optional)
        limit  : max results (optional, default 20)
        format : 'json' or 'text' (optional, default 'json')

    Response (json):
        {
            "results": [ { concept_label, core_data, promoted_date,
                           recall_count, unique_query_count, relevancy_score } ]
        }
    """
    q      = request.args.get('q', '').strip() or None
    tag    = request.args.get('tag', '').strip() or None
    limit  = int(request.args.get('limit', 20))
    fmt    = request.args.get('format', 'json')

    results = search_lore(text=q, tag=tag, limit=limit)

    if fmt == 'text':
        return format_lore_results(results), 200, {'Content-Type': 'text/plain'}

    return jsonify({
        'count':   len(results),
        'results': results,
    })


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"\n[SonglineMemory API] Starting on {HOST}:{PORT}")
    print(f"[SonglineMemory API] Key loaded from {ENV_PATH}")
    print(f"[SonglineMemory API] Endpoints: /health | /remember | /query | /lore\n")

    app.run(host=HOST, port=PORT, debug=False)
