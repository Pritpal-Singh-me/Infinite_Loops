#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Infinite Loop – Backend Startup Script (Fixed)
# ─────────────────────────────────────────────────────────────────────────────
# Usage:  bash start.sh           → fast mode (keyword NLP fallback)
#         bash start.sh --full-ai → full AI mode (Whisper + RawNet2 + AASIST)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "────────────────────────────────────────────────────────────"
echo "  ∞  Infinite Loop – Voice Cloning Detection Backend"
echo "────────────────────────────────────────────────────────────"

# ── 1. Find or create a virtual environment ──────────────────────────────────
# Priority: Downloads venv (already has most packages) > local venv > create new
DOWNLOADS_VENV="/Users/paragrajeshkumarsanghavi/Downloads/voiceguard-backend/venv"
LOCAL_VENV="$SCRIPT_DIR/venv"

if [ -f "$DOWNLOADS_VENV/bin/python" ]; then
    VENV_DIR="$DOWNLOADS_VENV"
    echo "→ Using existing venv: $DOWNLOADS_VENV"
elif [ -f "$LOCAL_VENV/bin/python" ]; then
    VENV_DIR="$LOCAL_VENV"
    echo "→ Using local venv: $LOCAL_VENV"
else
    echo "→ No venv found. Creating one at $LOCAL_VENV..."
    python3 -m venv "$LOCAL_VENV"
    VENV_DIR="$LOCAL_VENV"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_UVICORN="$VENV_DIR/bin/uvicorn"

# ── 2. Install core deps if missing ──────────────────────────────────────────
echo "→ Ensuring core packages are installed..."
"$VENV_PIP" install --quiet --upgrade \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    numpy \
    scipy \
    aiofiles \
    python-multipart \
    "fpdf2>=2.7.9" \
    "groq>=0.9.0" \
    pyyaml 2>&1 | tail -3

# ── 3. Optional full-AI stack ─────────────────────────────────────────────────
if [[ "$1" == "--full-ai" ]]; then
    echo "→ Installing full AI stack (librosa, whisper, torch, torchaudio)..."
    echo "   ⚠  This may take several minutes on first run."
    "$VENV_PIP" install --quiet librosa openai-whisper torch torchaudio 2>&1 | tail -5
fi

# ── 4. Check for .env (GROQ_API_KEY) ─────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo ""
    echo "  ⚠  WARNING: No .env file found at backend/.env"
    echo "     The backend will run in keyword-fallback mode (no Groq LLM)."
    echo ""
    echo "     To enable Groq LLM analysis, create backend/.env with:"
    echo "       GROQ_API_KEY=gsk_your_key_here"
    echo "     Get a free key at: https://console.groq.com"
    echo ""
else
    echo "→ Found .env file — Groq LLM will be enabled if key is valid"
fi

# ── 5. Check port 8000 ───────────────────────────────────────────────────────
if lsof -i :8000 > /dev/null 2>&1; then
    echo ""
    echo "  ℹ  Port 8000 is already in use."
    echo "     Checking if it's our backend..."
    HEALTH=$(curl -s --max-time 2 http://localhost:8000/health 2>/dev/null)
    if echo "$HEALTH" | grep -q '"status"'; then
        echo "  ✓  Backend is already running! Health: $HEALTH"
        echo ""
        echo "     WebSocket: ws://localhost:8000/ws/analyse"
        echo "     Docs:      http://localhost:8000/docs"
        exit 0
    else
        echo "  ✗  Port 8000 is used by a different process."
        echo "     Kill it with: lsof -ti:8000 | xargs kill -9"
        exit 1
    fi
fi

# ── 6. Launch server ──────────────────────────────────────────────────────────
echo ""
echo "  ✓ Backend starting at:   http://localhost:8000"
echo "  ✓ WebSocket endpoint:    ws://localhost:8000/ws/analyse"
echo "  ✓ Health check:          http://localhost:8000/health"
echo "  ✓ API docs:              http://localhost:8000/docs"
echo ""
echo "  Mode: $([ "$1" == "--full-ai" ] && echo "Full AI (Whisper + RawNet2 + AASIST)" || echo "Core mode (Groq LLM + keyword fallback)")"
echo ""
echo "  Press Ctrl+C to stop."
echo "────────────────────────────────────────────────────────────"

"$VENV_UVICORN" main:app --host 0.0.0.0 --port 8000 --reload
