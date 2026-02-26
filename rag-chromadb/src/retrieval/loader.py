"""Multi-source document loader"""
from typing import Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from src.retrieval import DocumentLoader
from src.models import SourceType
from src.utils import get_logger

logger = get_logger()


class MultiSourceDataLoader(DocumentLoader):
    """Load documents from multiple sources: Wikipedia, URLs, files, PDFs"""

    def __init__(self):
        self.loaded_sources = {}

    def load(self, source: str) -> str:
        """Load content from source (auto-detect type)"""
        # Check cache first
        if source in self.loaded_sources:
            logger.info(f"Loading from cache: {source}")
            return self.loaded_sources[source]

        source_type = self._detect_source_type(source)

        if source_type == SourceType.WIKIPEDIA:
            content = self._load_wikipedia(source)
        elif source_type == SourceType.URL:
            content = self._load_url(source)
        elif source_type == SourceType.PDF:
            content = self._load_pdf(source)
        else:
            content = self._load_file(source)

        self.loaded_sources[source] = content
        return content

    @staticmethod
    def _detect_source_type(source: str) -> str:
        """Detect source type from URL/path"""
        import os

        # Check URL schemes first
        if source.startswith("http://") or source.startswith("https://"):
            return SourceType.URL

        # Check Wikipedia keyword
        if "wikipedia" in source.lower():
            return SourceType.WIKIPEDIA

        # Check file extensions
        if source.endswith(".pdf"):
            return SourceType.PDF
        if source.endswith((".txt", ".md")):
            return SourceType.FILE

        # Check if file exists locally
        if os.path.isfile(source):
            return SourceType.FILE

        # Default to Wikipedia (assume it's a topic name)
        return SourceType.WIKIPEDIA

    @staticmethod
    def _load_wikipedia(page_title: str) -> str:
        """Load content from Wikipedia"""
        try:
            # Extract topic if it starts with "wikipedia "
            if page_title.lower().startswith("wikipedia "):
                page_title = page_title[10:]  # Remove "wikipedia "

            from wikipediaapi import Wikipedia
            USER_AGENT = "generative-ai-learning/1.0 (contact: your-email@example.com)"
            wiki = Wikipedia(user_agent=USER_AGENT, language="en")
            page = wiki.page(page_title)
            if page.exists():
                return page.text
            else:
                logger.warning(f"Wikipedia page not found: {page_title}")
                return ""
        except Exception as e:
            logger.error(f"Error loading Wikipedia: {e}")
            return ""

    @staticmethod
    def _load_url(url: str) -> str:
        """Load and scrape content from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            return "\n".join(line for line in lines if line)
        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            return ""

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """Load content from PDF file"""
        try:
            text = []
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return "\n".join(text)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""

    @staticmethod
    def _load_file(file_path: str) -> str:
        """Load content from text file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return ""
