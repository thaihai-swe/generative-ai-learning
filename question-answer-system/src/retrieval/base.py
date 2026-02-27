"""Base abstract classes for retrieval components"""
from abc import ABC, abstractmethod
from typing import List
from src.models import RetrievedDocument


class Retriever(ABC):
    """Abstract base retriever interface"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Retrieve documents matching query"""
        pass


class DocumentLoader(ABC):
    """Abstract document loader interface"""

    @abstractmethod
    def load(self, source: str) -> str:
        """Load content from source"""
        pass


class Chunker(ABC):
    """Abstract text chunker interface"""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks"""
        pass
