#!/bin/bash
# RAG System Development Runner
# This script runs the RAG system with the correct Python environment

VENV_PATH="./venv/bin/python3"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if venv exists
if [ ! -f "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Run the main application
"$VENV_PATH" "$SCRIPT_DIR/main.py" "$@"
