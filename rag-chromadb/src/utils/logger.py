"""Logging utilities"""
import logging
import os
from typing import Optional


class RAGLogger:
    """Centralized logging for RAG system"""

    _instance: Optional["RAGLogger"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.logger = logging.getLogger("rag-system")
        self._configure()
        self._initialized = True

    def _configure(self):
        """Configure logger with handlers and formatters"""
        # Remove any existing handlers
        self.logger.handlers.clear()

        # Only configure if not already configured
        if self.logger.hasHandlers():
            return

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Suppress ChromaDB telemetry
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(
            logging.CRITICAL
        )
        logging.getLogger("chromadb").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

    @staticmethod
    def get_logger() -> logging.Logger:
        """Get or create logger instance"""
        logger = RAGLogger()
        return logger.logger


def get_logger() -> logging.Logger:
    """Convenience function to get logger"""
    return RAGLogger.get_logger()
