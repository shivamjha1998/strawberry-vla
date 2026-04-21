#!/bin/bash
set -e

cd "$(dirname "$0")"

# Activate venv if present
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Kill any leftover process holding port 7860
EXISTING=$(lsof -ti tcp:7860 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Freeing port 7860..."
    kill $EXISTING 2>/dev/null || true
    sleep 1
fi

echo "Starting Strawberry VLA demo..."
python demo_app.py --preload &
APP_PID=$!

# Wait for Gradio to be ready
echo "Waiting for app to start..."
until curl -s http://localhost:7860 > /dev/null 2>&1; do
    sleep 1
done

echo ""
echo "App is running. Starting Cloudflare tunnel..."
echo ""

# Run tunnel in foreground so the URL is visible; Ctrl+C stops everything
trap "kill $APP_PID 2>/dev/null; exit 0" INT TERM
cloudflared tunnel --url http://localhost:7860

kill $APP_PID 2>/dev/null
