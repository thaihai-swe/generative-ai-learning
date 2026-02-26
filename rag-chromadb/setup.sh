#!/bin/bash
# Setup script for RAG System
# This script installs all dependencies in the venv

echo "🔧 Setting up RAG System..."
echo ""

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate venv and install requirements
echo "📦 Installing dependencies..."
./venv/bin/pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Setup complete!"
    echo ""
    echo "To run the system:"
    echo "  ./run.sh"
    echo ""
    echo "Or manually:"
    echo "  source venv/bin/activate"
    echo "  python main.py"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
