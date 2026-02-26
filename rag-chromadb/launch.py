#!/usr/bin/env python3
"""RAG System Launcher - Setup and Run"""
import sys
import os
import subprocess

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, "venv", "bin", "python3")
    
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found")
        print("🔧 Setting up environment...")
        return 1
    
    print("🚀 Starting RAG System...")
    try:
        subprocess.run([venv_python, "main.py"], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
