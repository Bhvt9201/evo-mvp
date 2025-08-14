#!/bin/bash
set -e

echo "=== Evo-MVP Setup Script ==="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew found"
fi

# Install Python 3.12 if not installed
if ! brew list python@3.12 &>/dev/null; then
    echo "Installing Python 3.12..."
    brew install python@3.12
else
    echo "✅ Python 3.12 already installed"
fi

PYTHON_PATH=$(brew --prefix python@3.12)/bin/python3.12

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    $PYTHON_PATH -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip & wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "❌ No requirements.txt found"
    exit 1
fi

# Check if port 5000 is in use
PORT=5000
while lsof -i :$PORT >/dev/null 2>&1; do
    echo "⚠️  Port $PORT is in use. Trying next..."
    PORT=$((PORT + 1))
done

echo "✅ Using port $PORT"

# Run app on the available port
echo "Running Flask app..."
FLASK_APP=app.py flask run --port=$PORT
