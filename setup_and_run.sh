#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_and_run.sh  —  FDC-AI project setup for macOS (Apple Silicon M4)
# Usage:  bash setup_and_run.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

PROJECT="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT/.venv"

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║   FDC-AI — Fault Detection Pipeline Setup     ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# 1. Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "❌  Python3 not found. Install via: brew install python"
    exit 1
fi
PV=$(python3 --version)
echo "✅  $PV"

# 2. Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "📦  Creating virtual environment..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# 3. Install packages ──────────────────────────────────────────────────────────
echo "📥  Installing packages (first run ~3–5 min, PyTorch is large)..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT/requirements.txt"
echo "✅  All packages installed"

# 4. Create directories ────────────────────────────────────────────────────────
mkdir -p "$PROJECT/data" "$PROJECT/models" "$PROJECT/static"

# 5. Train pipeline ────────────────────────────────────────────────────────────
echo ""
echo "🔧  Training FDC pipeline (Isolation Forest + Autoencoder + XGBoost)..."
echo "    This takes 2–4 minutes on M4."
echo ""
cd "$PROJECT"
python train.py

# 6. Launch API ────────────────────────────────────────────────────────────────
echo ""
echo "🚀  Starting FastAPI server..."
echo "    Dashboard:  http://localhost:8000"
echo "    Swagger UI: http://localhost:8000/docs"
echo "    Press Ctrl+C to stop."
echo ""
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
