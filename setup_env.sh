#!/usr/bin/env bash
# ============================================================================
# MahaMonsoon-X Unix/macOS/WSL Setup Script (Python 3.12.10 required)
# ============================================================================

set -e

REQUIRED_PY="3.12.10"
echo "[INFO] Initializing MahaMonsoon-X Environment Setup..."
echo "[INFO] Target Python Version: ${REQUIRED_PY}"

# Check for Python 3.12
PY_CMD=""
if command -v python3.12 &> /dev/null; then
    PY_CMD="python3.12"
elif command -v python3 &> /dev/null && python3 -c 'import sys; exit(0 if sys.version_info[:2] == (3, 12) else 1)' &> /dev/null; then
    PY_CMD="python3"
else
    echo "[ERROR] Python 3.12.10 is not installed or not in PATH."
    echo "[ACTION] Please install Python 3.12.10 using pyenv or your package manager:"
    echo "         pyenv install 3.12.10"
    echo "         pyenv local 3.12.10"
    exit 1
fi

echo "[OK] Using Python binary: $($PY_CMD -V)"

if [ ! -d ".venv312" ]; then
    echo "[INFO] Creating virtual environment at .venv312..."
    $PY_CMD -m venv .venv312
else
    echo "[OK] Existing virtual environment found at .venv312."
fi

echo "[INFO] Activating virtual environment..."
source .venv312/bin/activate

echo "[INFO] Upgrading pip and core packaging tools..."
pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

echo ""
echo "============================================================================"
echo "[SUCCESS] MahaMonsoon-X environment is fully initialized!"
python -c "import sys; print('Active Python:', sys.executable, sys.version)"
echo ""
echo "To activate:"
echo "  source .venv312/bin/activate"
echo "============================================================================"
