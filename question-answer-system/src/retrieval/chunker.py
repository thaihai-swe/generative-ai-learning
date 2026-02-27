"""Adaptive chunking strategy for various content types"""
from abc import ABC
from typing import List
import re
from src.retrieval import Chunker
from src.models import ContentType


class AdaptiveChunker(Chunker):
    """Content-aware chunking strategy that adapts based on content type"""

    def __init__(
        self,
        default_chunk_size: int = 512,
        default_overlap: int = 50,
    ):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    def chunk(self, text: str) -> List[str]:
        """Adaptively chunk based on content type"""
        content_type = self._detect_content_type(text)
        chunk_size, overlap = self._get_optimal_params(content_type)
        return self._chunk_with_overlap(text, chunk_size, overlap)

    @staticmethod
    def _detect_content_type(text: str) -> str:
        """Detect content type based on text characteristics"""
        # Academic: citations, equations, formal structure
        if re.search(r"\[.*\]|\(.*et al\.|\\[a-z]+|^\d+\.", text, re.IGNORECASE):
            return "academic"
        # Structured: tables, lists, clear hierarchy
        if re.search(r"^\s*[-*•]\s|^\d+\.|^#{1,6}\s", text, re.MULTILINE):
            return "structured"
        return "general"

    def _get_optimal_params(self, content_type: str) -> tuple[int, int]:
        """Get optimal chunk size and overlap for content type"""
        if content_type == "academic":
            return (1024, 200)  # Larger chunks, more overlap
        elif content_type == "structured":
            return (256, 32)  # Smaller chunks, less overlap
        else:
            return (512, 50)  # Default

    @staticmethod
    def _chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
