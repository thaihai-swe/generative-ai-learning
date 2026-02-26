"""Enumeration types for RAG system"""
from enum import Enum


class SourceType(str, Enum):
    """Source type enumeration"""
    WIKIPEDIA = "wikipedia"
    URL = "url"
    FILE = "file"
    PDF = "pdf"


class RetrievalMethod(str, Enum):
    """Retrieval method"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class ContentType(str, Enum):
    """Content type for adaptive chunking"""
    ACADEMIC = "academic"
    STRUCTURED = "structured"
    GENERAL = "general"
