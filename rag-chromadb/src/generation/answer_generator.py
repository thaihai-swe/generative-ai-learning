from abc import ABC, abstractmethod
from typing import List, Iterator
from src.models import RetrievedDocument
from src.config import get_config
from src.utils import get_logger
from openai import OpenAI

logger = get_logger()


class AnswerGenerator(ABC):
    """Abstract answer generator interface"""

    @abstractmethod
    def generate(self, query: str, context: List[RetrievedDocument]) -> str:
        """Generate answer from context"""
        pass

    @abstractmethod
    def generate_streaming(self, query: str, context: List[RetrievedDocument]) -> Iterator[str]:
        """Stream answer token-by-token"""
        pass
