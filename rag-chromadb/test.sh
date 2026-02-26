#!/bin/bash
# Quick test to verify RAG system is working

echo "🧪 Testing RAG System..."
echo ""

VENV_PYTHON="./venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Run import tests
"$VENV_PYTHON" << 'EOF'
print("Testing imports...")
try:
    import chromadb
    print("  ✅ chromadb")
    import openai
    print("  ✅ openai")
    from src.config import get_config
    print("  ✅ src.config")
    from src.models import RetrievedDocument
    print("  ✅ src.models")
    from src.retrieval import Retriever
    print("  ✅ src.retrieval")
    from src.generation import LLMAnswerGenerator
    print("  ✅ src.generation")
    from src.evaluation import RAGASEvaluator
    print("  ✅ src.evaluation")
    from src.core import EnhancedRAGSystem
    print("  ✅ src.core")
    from src.cli import InteractiveRAG
    print("  ✅ src.cli")
    print("")
    print("✅ All tests passed! System is ready.")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

exit $?
