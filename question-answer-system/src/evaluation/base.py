"""Base evaluator abstract class"""
from abc import ABC, abstractmethod
from src.models import RAGASMetrics


class Evaluator(ABC):
    """Abstract evaluator interface"""

    @abstractmethod
    def evaluate(self, query: str, context: str, answer: str) -> RAGASMetrics:
        """Evaluate RAG output quality"""
        pass
