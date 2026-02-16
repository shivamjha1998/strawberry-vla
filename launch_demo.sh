#!/bin/bash
# launch_demo.sh — Start the Strawberry VLA Demo
#
# Usage:
#   ./launch_demo.sh           # Normal launch (model loads on first request)
#   ./launch_demo.sh --preload # Pre-load model before starting UI
#   ./launch_demo.sh --share   # Create a public URL for remote demo

cd "$(dirname "$0")"
source venv/bin/activate

echo "=============================================="
echo "🍓 Strawberry VLA — Starting Demo"
echo "=============================================="
echo ""
echo "  URL: http://localhost:7860"
echo "  Press Ctrl+C to stop"
echo ""

python demo_app.py "$@"
