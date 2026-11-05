#!/bin/bash
# Kwaya — Choral Score Assistant
# Start the FastAPI server (uses python 3.12 which has all deps)

cd "$(dirname "$0")"
echo "🎵 Kwaya — Choral Score Assistant"
echo "═══════════════════════════════════"
echo ""

PORT=${KWAYA_PORT:-8742}

echo "🚀 Starting server on http://localhost:$PORT"
echo "   Open this URL in your browser"
echo "   Press Ctrl+C to stop"
echo ""

# Use 'python' not 'python3' — python=3.12 has music21 + all deps
python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT