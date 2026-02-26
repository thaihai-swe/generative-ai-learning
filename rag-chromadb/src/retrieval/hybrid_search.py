"""Hybrid search engine combining semantic and keyword search"""
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from src.retrieval import Retriever
from src.models import RetrievedDocument
from src.config import get_config
from src.utils import get_logger

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = get_logger()


class HybridSearchEngine(Retriever):
    """Implements hybrid search combining BM25 keyword and semantic search"""

    def __init__(self, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        self.chunk_storage: Dict[str, List[str]] = {}
        self.stop_words = set(stopwords.words('english'))
        self.chroma_client = None
        self.embedding_function = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text"""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in self.stop_words]

    def build_bm25_index(self, collection_name: str, documents: List[str]) -> None:
        """Build BM25 index for keyword search"""
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25_indices[collection_name] = BM25Okapi(tokenized_docs)
        self.chunk_storage[collection_name] = documents
        logger.info(f"✅ Built BM25 index for {collection_name} with {len(documents)} documents")

    def keyword_search(
        self, collection_name: str, query: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search"""
        if collection_name not in self.bm25_indices:
            logger.warning(f"⚠️ No BM25 index for {collection_name}")
            return []

        query_tokens = self._tokenize(query)
        bm25 = self.bm25_indices[collection_name]
        scores = bm25.get_scores(query_tokens)

        # Get top-k results with scores
        ranked = sorted(
            enumerate(zip(self.chunk_storage[collection_name], scores)),
            key=lambda x: x[1][1],
            reverse=True
        )[:top_k]

        return [(doc, score) for idx, (doc, score) in ranked]

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        max_score = max(scores)
        if max_score == 0:
            return scores
        return [score / max_score for score in scores]

    def _combine_results(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Combine semantic and keyword search results with weighted ensemble"""
        combined: Dict[str, float] = {}

        # Add semantic results
        if semantic_results:
            semantic_scores = self._normalize_scores([score for _, score in semantic_results])
            for i, (doc, _) in enumerate(semantic_results):
                score = semantic_scores[i] * self.semantic_weight
                combined[doc] = combined.get(doc, 0) + score

        # Add keyword results
        if keyword_results:
            keyword_scores = self._normalize_scores([score for _, score in keyword_results])
            for i, (doc, _) in enumerate(keyword_results):
                score = keyword_scores[i] * self.keyword_weight
                combined[doc] = combined.get(doc, 0) + score

        # Return sorted by combined score
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Retrieve documents using hybrid search"""
        # This is a placeholder - actual implementation depends on ChromaDB setup
        logger.info(f"🔍 Hybrid search for: {query} (top_k={top_k})")
        return []
