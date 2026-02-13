import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pprint import pprint
from urllib.parse import urlparse
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from wikipediaapi import Wikipedia
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tabulate import tabulate

# Download NLTK data for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load environment variables
load_dotenv()

# Configuration
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "lm-studio")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
MAX_TOOL_CALLS = 10
MAX_RETRIEVED_CHUNKS = 3
CONVERSATION_HISTORY_FILE = "./conversation_history.json"
EVALUATION_METRICS_FILE = "./evaluation_metrics.json"
HYBRID_SEARCH_WEIGHT_SEMANTIC = 0.7  # 70% semantic, 30% keyword
HYBRID_SEARCH_WEIGHT_KEYWORD = 0.3

# Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

client = OpenAI(base_url=OPEN_AI_API_BASE_URL, api_key=OPEN_AI_API_KEY)


# Data Models
@dataclass
class RetrievedDocument:
    """Represents a retrieved document chunk with metadata."""
    content: str
    source: str
    source_type: str  # 'wikipedia', 'url', 'file'
    index: int
    distance: Optional[float] = None

    @property
    def relevance_score(self) -> float:
        """Convert distance to relevance score (0-1, higher is better)."""
        if self.distance is None:
            return 0.5
        return max(0, 1 - (self.distance / 2))


@dataclass
class ConversationMessage:
    """Represents a message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[Dict]] = None
    confidence_score: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class RAGResponse:
    """Structured RAG response with metadata."""
    answer: str
    sources: List[RetrievedDocument]
    confidence_score: float
    source_types: List[str]
    conversation_context: str


@dataclass
class RAGASMetrics:
    """RAGAS evaluation metrics for RAG quality assessment."""
    context_relevance: float  # Are retrieved docs relevant? (0-1)
    answer_relevance: float   # Does answer address question? (0-1)
    faithfulness: float       # Is answer grounded in context? (0-1)
    rag_score: float          # Overall RAG quality (0-1)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return f"""RAGAS Metrics:
  Context Relevance:  {self.context_relevance:.1%}
  Answer Relevance:   {self.answer_relevance:.1%}
  Faithfulness:       {self.faithfulness:.1%}
  ─────────────────────────
  Overall RAG Score:  {self.rag_score:.1%}"""


@dataclass
class EvaluationResult:
    """Full evaluation result for a query-answer pair."""
    query: str
    answer: str
    metrics: RAGASMetrics
    retrieval_method: str  # 'semantic', 'keyword', 'hybrid'
    num_chunks_retrieved: int
    timestamp: str

    def to_dict(self):
        return {
            'query': self.query,
            'answer': self.answer,
            'metrics': self.metrics.to_dict(),
            'retrieval_method': self.retrieval_method,
            'num_chunks_retrieved': self.num_chunks_retrieved,
            'timestamp': self.timestamp
        }




db_client = chromadb.PersistentClient(path="./chroma_db")
db_client.heartbeat()
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Set a proper user agent for Wikipedia API
USER_AGENT = "generative-ai-learning/1.0 (contact: your-email@example.com)"
wiki = Wikipedia(user_agent=USER_AGENT, language="en")


class AdaptiveChunker:
    """Intelligently chunks text based on content characteristics."""

    @staticmethod
    def detect_content_type(text: str) -> str:
        """Detect content type to optimize chunk size."""
        # Simple heuristics for content type detection
        lines = text.split('\n')
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)

        # Check for academic indicators
        academic_keywords = ['research', 'study', 'analysis', 'methodology', 'abstract', 'conclusion']
        academic_count = sum(1 for keyword in academic_keywords if keyword.lower() in text.lower()[:500])

        if academic_count >= 2:
            return 'academic'
        elif avg_line_length < 60:
            return 'structured'  # Code, formatted text
        else:
            return 'general'

    @staticmethod
    def get_optimal_chunk_size(content_type: str) -> Tuple[int, int]:
        """Get optimal chunk size and overlap for content type."""
        chunk_configs = {
            'academic': (800, 200),      # Large chunks, good overlap
            'structured': (300, 50),     # Smaller chunks, minimal overlap
            'general': (500, 100)        # Medium chunks
        }
        return chunk_configs.get(content_type, (500, 100))

    @staticmethod
    def chunk_with_overlap(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @staticmethod
    def adaptive_chunk(text: str) -> List[str]:
        """Adaptively chunk text based on content characteristics."""
        content_type = AdaptiveChunker.detect_content_type(text)
        chunk_size, overlap = AdaptiveChunker.get_optimal_chunk_size(content_type)

        logger.info(f"🔍 Detected content type: {content_type} (chunk_size={chunk_size}, overlap={overlap})")

        return AdaptiveChunker.chunk_with_overlap(text, chunk_size, overlap)


class HybridSearchEngine:
    """Implements hybrid search combining BM25 keyword and semantic search."""

    def __init__(self):
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        self.chunk_storage: Dict[str, List[str]] = {}  # Store chunks for BM25
        self.stop_words = set(stopwords.words('english'))

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text."""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in self.stop_words]

    def build_bm25_index(self, collection_name: str, documents: List[str]):
        """Build BM25 index for keyword search."""
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25_indices[collection_name] = BM25Okapi(tokenized_docs)
        self.chunk_storage[collection_name] = documents
        logger.debug(f"✅ Built BM25 index for {collection_name} with {len(documents)} documents")

    def keyword_search(self, collection_name: str, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search."""
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

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return scores
        max_score = max(scores)
        if max_score == 0:
            return scores
        return [score / max_score for score in scores]

    def hybrid_search(self,
                     query: str,
                     semantic_results: List[Tuple[str, float]],
                     keyword_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine semantic and keyword search results with weighted ensemble."""
        combined = {}

        # Add semantic results
        semantic_scores = self.normalize_scores([score for _, score in semantic_results])
        for i, (doc, _) in enumerate(semantic_results):
            score = semantic_scores[i] * HYBRID_SEARCH_WEIGHT_SEMANTIC
            combined[doc] = combined.get(doc, 0) + score

        # Add keyword results
        keyword_scores = self.normalize_scores([score for _, score in keyword_results])
        for i, (doc, _) in enumerate(keyword_results):
            score = keyword_scores[i] * HYBRID_SEARCH_WEIGHT_KEYWORD
            combined[doc] = combined.get(doc, 0) + score

        # Return sorted by combined score
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)


class RAGEvaluator:
    """Evaluates RAG quality using RAGAS-inspired metrics."""

    @staticmethod
    def evaluate_context_relevance(query: str, context: str) -> float:
        """
        Context Relevance: Evaluates if retrieved context is relevant to the query.
        Uses LLM to score relevance.
        """
        try:
            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating the relevance of document context to a query. Respond with only a number from 0 to 10."
                    },
                    {
                        "role": "user",
                        "content": f"""Query: {query}

Context: {context[:500]}

On a scale of 0-10, how relevant is this context to the query?
Only provide a number."""
                    }
                ],
                temperature=0.3,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            score = float(''.join(filter(str.isdigit, score_text.split('\n')[0]))) / 10.0
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"⚠️ Context relevance eval failed: {str(e)}")
            return 0.5

    @staticmethod
    def evaluate_answer_relevance(query: str, answer: str) -> float:
        """
        Answer Relevance: Evaluates if the answer addresses the query.
        Uses LLM to score answer directly addressing the question.
        """
        try:
            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating if an answer directly addresses a query. Respond with only a number from 0 to 10."
                    },
                    {
                        "role": "user",
                        "content": f"""Query: {query}

Answer: {answer[:500]}

On a scale of 0-10, how well does this answer address the query?
Only provide a number."""
                    }
                ],
                temperature=0.3,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            score = float(''.join(filter(str.isdigit, score_text.split('\n')[0]))) / 10.0
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"⚠️ Answer relevance eval failed: {str(e)}")
            return 0.5

    @staticmethod
    def evaluate_faithfulness(context: str, answer: str) -> float:
        """
        Faithfulness: Evaluates if the answer is grounded in the provided context.
        Checks for hallucinations by verifying answer against context.
        """
        try:
            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating if an answer is grounded in provided context. Respond with only a number from 0 to 10."
                    },
                    {
                        "role": "user",
                        "content": f"""Context: {context[:500]}

Answer: {answer[:500]}

On a scale of 0-10, how much of the answer is supported by the context?
Only provide a number."""
                    }
                ],
                temperature=0.3,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            score = float(''.join(filter(str.isdigit, score_text.split('\n')[0]))) / 10.0
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"⚠️ Faithfulness eval failed: {str(e)}")
            return 0.5

    @staticmethod
    def compute_rag_score(context_relevance: float,
                          answer_relevance: float,
                          faithfulness: float) -> float:
        """Compute overall RAG score as weighted average."""
        # Weights: Context (30%), Answer (35%), Faithfulness (35%)
        weights = [0.30, 0.35, 0.35]
        scores = [context_relevance, answer_relevance, faithfulness]
        return sum(s * w for s, w in zip(scores, weights))

    @staticmethod
    def evaluate(query: str, context: str, answer: str) -> RAGASMetrics:
        """Perform full RAGAS evaluation."""
        logger.info("📊 Running RAGAS evaluation...")

        context_relevance = RAGEvaluator.evaluate_context_relevance(query, context)
        answer_relevance = RAGEvaluator.evaluate_answer_relevance(query, answer)
        faithfulness = RAGEvaluator.evaluate_faithfulness(context, answer)
        rag_score = RAGEvaluator.compute_rag_score(context_relevance, answer_relevance, faithfulness)

        return RAGASMetrics(
            context_relevance=context_relevance,
            answer_relevance=answer_relevance,
            faithfulness=faithfulness,
            rag_score=rag_score
        )


class MultiSourceDataLoader:
    """Handles loading data from multiple sources (Wikipedia, URLs, etc.)"""

    @staticmethod
    def scrape_url(url: str) -> Optional[str]:
        """Scrape content from a URL using BeautifulSoup."""
        logger.info(f"🌐 Scraping URL: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

            if text:
                logger.info(f"✅ Successfully scraped {len(text)} characters from URL")
                return text
            else:
                logger.warning(f"⚠️ No content found on {url}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to scrape URL: {str(e)}")
            return None

    @staticmethod
    def load_wikipedia_page(page_name: str) -> Optional[str]:
        """Load content from Wikipedia."""
        logger.info(f"📚 Loading Wikipedia page: {page_name}")
        try:
            page = wiki.page(page_name)

            if not page.exists():
                logger.warning(f"⚠️ Wikipedia page '{page_name}' not found")
                return None

            text = page.text
            if text:
                logger.info(f"✅ Retrieved {len(text)} characters from Wikipedia")
                return text
            return None

        except Exception as e:
            logger.error(f"❌ Wikipedia retrieval failed: {str(e)}")
            return None

    @staticmethod
    def load_file(file_path: str) -> Optional[str]:
        """Load content from a local text file."""
        logger.info(f"📄 Loading file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if text:
                logger.info(f"✅ Retrieved {len(text)} characters from file")
                return text
            return None

        except Exception as e:
            logger.error(f"❌ File loading failed: {str(e)}")
            return None

    @staticmethod
    def detect_source_type(source: str) -> str:
        """Detect the type of source (wikipedia, url, or file)."""
        if source.startswith(('http://', 'https://')):
            return 'url'
        elif source.endswith(('.txt', '.md', '.pdf')):
            return 'file'
        else:
            return 'wikipedia'

    @staticmethod
    def load_from_source(source: str) -> Tuple[Optional[str], str]:
        """Load content from any source, returns (content, source_type)."""
        source_type = MultiSourceDataLoader.detect_source_type(source)

        if source_type == 'url':
            content = MultiSourceDataLoader.scrape_url(source)
        elif source_type == 'file':
            content = MultiSourceDataLoader.load_file(source)
        else:
            content = MultiSourceDataLoader.load_wikipedia_page(source)

        return content, source_type


class EnhancedRAGSystem:
    """Advanced RAG system with multi-source support and conversation history."""

    def __init__(self):
        self.data_loader = MultiSourceDataLoader()
        self.collections: Dict[str, chromadb.Collection] = {}
        self.conversation_history: List[ConversationMessage] = []
        self.conversation_id = self._generate_conversation_id()
        self.loaded_sources: Dict[str, str] = {}  # source_name -> source_type
        self.evaluation_results: List[EvaluationResult] = []
        self.hybrid_engine = HybridSearchEngine()  # New: hybrid search engine
        self.evaluator = RAGEvaluator()            # New: RAG evaluator
        self._load_conversation_history()
        logger.info(f"✅ Initialized RAG System with conversation ID: {self.conversation_id}")
        logger.info(f"✅ Hybrid Search Engine initialized")
        logger.info(f"✅ RAGAS Evaluator initialized")

    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_collection_name(self, source: str) -> str:
        """Generate sanitized collection name from source."""
        parsed = urlparse(source)
        if parsed.netloc:
            name = parsed.netloc.replace('.', '_').replace('/', '_')
        else:
            name = source.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return name.lower()[:50]  # ChromaDB has limits on collection name length

    def load_source(self, source: str) -> bool:
        """Load content from a source and store in ChromaDB."""
        logger.info(f"📖 Loading source: {source}")

        content, source_type = self.data_loader.load_from_source(source)

        if not content:
            print(f"❌ Failed to load content from {source}")
            return False

        collection_name = self._get_collection_name(source)

        try:
            # Use adaptive chunking based on content type
            chunks = AdaptiveChunker.adaptive_chunk(content)

            if not chunks:
                print(f"❌ No valid chunks extracted from {source}")
                return False

            # Create or get collection
            collection = db_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )

            # Add documents with metadata
            collection.add(
                ids=[f"{collection_name}_{i}" for i in range(len(chunks))],
                documents=chunks,
                metadatas=[
                    {
                        "source": source,
                        "source_type": source_type,
                        "index": i,
                        "timestamp": datetime.now().isoformat()
                    } for i in range(len(chunks))
                ]
            )

            # Build BM25 index for hybrid search (NEW)
            self.hybrid_engine.build_bm25_index(collection_name, chunks)

            self.collections[collection_name] = collection
            self.loaded_sources[source] = source_type

            print(f"✅ Successfully loaded {len(chunks)} chunks from {source}")
            print(f"   Source Type: {source_type.upper()}")
            print(f"   Collection: {collection_name}")
            print(f"   Chunking Strategy: Adaptive\n")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to store in ChromaDB: {str(e)}")
            print(f"❌ Error: {str(e)}")
            return False

    def _retrieve_relevant_chunks(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS, use_hybrid: bool = True) -> Tuple[List[RetrievedDocument], str]:
        """Retrieve relevant chunks using semantic or hybrid search."""
        logger.info(f"🔍 Retrieving chunks for query: {query[:50]}...")

        all_results = []
        retrieval_method = "semantic"

        for collection_name, collection in self.collections.items():
            try:
                if use_hybrid and collection_name in self.hybrid_engine.bm25_indices:
                    # Hybrid search: combine semantic and keyword searches
                    retrieval_method = "hybrid"
                    logger.info(f"  Using HYBRID search for {collection_name}")

                    # Semantic search
                    semantic_results = collection.query(
                        query_texts=[query],
                        n_results=min(n_results, MAX_RETRIEVED_CHUNKS),
                        include=["documents", "metadatas", "distances"]
                    )

                    semantic_docs = []
                    if semantic_results['documents'] and semantic_results['documents'][0]:
                        for i, doc in enumerate(semantic_results['documents'][0]):
                            distance = semantic_results['distances'][0][i] if semantic_results['distances'] else None
                            # Score: 1 - distance (normalized to 0-1)
                            score = max(0, 1 - (distance / 2)) if distance is not None else 0.5
                            semantic_docs.append((doc, score))

                    # Keyword search
                    keyword_docs = self.hybrid_engine.keyword_search(collection_name, query, n_results)

                    # Combine results
                    combined_docs = self.hybrid_engine.hybrid_search(query, semantic_docs, keyword_docs)

                    # Convert to RetrievedDocument objects
                    for doc, combined_score in combined_docs:
                        metadata = {"source": self.loaded_sources.get(list(self.loaded_sources.keys())[0], "Unknown"),
                                  "source_type": "unknown", "index": 0}
                        # Find metadata from collection
                        for chunk_idx, chunk_doc in enumerate(self.hybrid_engine.chunk_storage[collection_name]):
                            if chunk_doc == doc:
                                # Get metadata from collection
                                try:
                                    result = collection.query(query_texts=[doc], include=["metadatas"])
                                    if result['metadatas'] and result['metadatas'][0]:
                                        metadata = result['metadatas'][0][0]
                                except:
                                    pass
                                break

                        all_results.append(RetrievedDocument(
                            content=doc,
                            source=metadata.get('source', 'Unknown'),
                            source_type=metadata.get('source_type', 'unknown'),
                            index=metadata.get('index', 0),
                            distance=1 - combined_score  # Convert back to distance
                        ))

                else:
                    # Semantic search only
                    logger.info(f"  Using SEMANTIC search for {collection_name}")
                    results = collection.query(
                        query_texts=[query],
                        n_results=min(n_results, MAX_RETRIEVED_CHUNKS),
                        include=["documents", "metadatas", "distances"]
                    )

                    if results['documents'] and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0]):
                            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                            distance = results['distances'][0][i] if results['distances'] else None

                            all_results.append(RetrievedDocument(
                                content=doc,
                                source=metadata.get('source', 'Unknown'),
                                source_type=metadata.get('source_type', 'unknown'),
                                index=metadata.get('index', 0),
                                distance=distance
                            ))
            except Exception as e:
                logger.warning(f"⚠️ Query failed for collection {collection_name}: {str(e)}")

        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        top_results = all_results[:MAX_RETRIEVED_CHUNKS]

        logger.info(f"✅ Retrieved {len(top_results)} relevant chunks using {retrieval_method}")
        return top_results, retrieval_method

    def _build_conversation_context(self, max_messages: int = 4) -> str:
        """Build context from recent conversation history."""
        if not self.conversation_history:
            return ""

        recent_messages = self.conversation_history[-max_messages:]
        context = ""

        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context += f"{role}: {msg.content[:150]}...\n"

        return context

    def _generate_answer(self, query: str, context_docs: List[RetrievedDocument]) -> Tuple[str, float]:
        """Generate answer using RAG with conversation context."""
        logger.info(f"🤖 Generating answer...")

        # Build context from retrieved documents
        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += f"\n[Source {i} - {doc.source_type.upper()}]\n{doc.content}\n"

        # Build conversation context
        conv_context = self._build_conversation_context()

        if not context:
            return "I don't have enough information to answer your question.", 0.3

        # Calculate average confidence
        avg_confidence = sum(doc.relevance_score for doc in context_docs) / len(context_docs) if context_docs else 0.5

        system_prompt = """You are a knowledgeable assistant that provides accurate answers based on provided context.
Guidelines:
- Only use information from the provided context
- If the answer is not in the context, clearly state: "Based on the provided sources, I don't have information about this"
- Cite which source you're using
- Be precise, well-structured, and concise
- If this is a follow-up question, acknowledge the previous context"""

        user_content = f"""Previous Conversation Context:
{conv_context}

Retrieved Context from Sources:
{context}

User Question: {query}

Please provide a clear, accurate answer based on the context and sources provided."""

        try:
            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            logger.info(f"✅ Answer generated (confidence: {avg_confidence:.2f})")
            return answer, avg_confidence

        except Exception as e:
            logger.error(f"❌ Answer generation failed: {str(e)}")
            return f"Error generating answer: {str(e)}", 0.0

    def process_query(self, user_query: str, enable_evaluation: bool = True) -> Tuple[RAGResponse, Optional[RAGASMetrics]]:
        """Process query through the RAG pipeline with evaluation."""
        logger.info("=" * 80)
        logger.info(f"🚀 Processing query: {user_query[:60]}...")

        try:
            # Step 1: Retrieve relevant chunks using hybrid search
            retrieved_docs, retrieval_method = self._retrieve_relevant_chunks(user_query, use_hybrid=True)

            # Step 2: Generate answer
            answer, confidence = self._generate_answer(user_query, retrieved_docs)

            # Step 3: Build context for evaluation
            context = "\n".join([doc.content for doc in retrieved_docs])

            # Step 4: Evaluate RAG quality (NEW)
            rag_metrics = None
            if enable_evaluation:
                logger.info("📊 Evaluating RAG quality using RAGAS...")
                rag_metrics = self.evaluator.evaluate(user_query, context, answer)
                logger.info(f"✅ Evaluation complete:\n{rag_metrics}")

                # Store evaluation result
                eval_result = EvaluationResult(
                    query=user_query,
                    answer=answer,
                    metrics=rag_metrics,
                    retrieval_method=retrieval_method,
                    num_chunks_retrieved=len(retrieved_docs),
                    timestamp=datetime.now().isoformat()
                )
                self.evaluation_results.append(eval_result)
                self._save_evaluation_metrics()

            # Extract source info
            sources = [doc.source for doc in retrieved_docs]
            source_types = list(set([doc.source_type for doc in retrieved_docs]))

            # Store in conversation history
            self.conversation_history.append(ConversationMessage(
                role="user",
                content=user_query,
                timestamp=datetime.now().isoformat(),
                sources=[{"source": doc.source, "type": doc.source_type} for doc in retrieved_docs]
            ))

            self.conversation_history.append(ConversationMessage(
                role="assistant",
                content=answer,
                timestamp=datetime.now().isoformat(),
                confidence_score=confidence,
                sources=[{"source": doc.source, "type": doc.source_type} for doc in retrieved_docs]
            ))

            # Save conversation history
            self._save_conversation_history()

            logger.info(f"✅ Query processed successfully using {retrieval_method} search")
            logger.info("=" * 80)

            return RAGResponse(
                answer=answer,
                sources=retrieved_docs,
                confidence_score=confidence,
                source_types=source_types,
                conversation_context=self._build_conversation_context()
            ), rag_metrics

        except Exception as e:
            logger.error(f"❌ Query processing failed: {str(e)}")
            return RAGResponse(
                answer=f"Error: {str(e)}",
                sources=[],
                confidence_score=0.0,
                source_types=[],
                conversation_context=""
            ), None

    def _save_conversation_history(self):
        """Save conversation history to file."""
        try:
            history_data = {
                "conversation_id": self.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "messages": [asdict(msg) if hasattr(msg, '__dataclass_fields__') else msg.to_dict() for msg in self.conversation_history]
            }

            with open(CONVERSATION_HISTORY_FILE, 'w') as f:
                json.dump(history_data, f, indent=2)

            logger.debug("✅ Conversation history saved")
        except Exception as e:
            logger.error(f"❌ Failed to save conversation history: {str(e)}")

    def _load_conversation_history(self):
        """Load conversation history from file."""
        try:
            if os.path.exists(CONVERSATION_HISTORY_FILE):
                with open(CONVERSATION_HISTORY_FILE, 'r') as f:
                    data = json.load(f)

                for msg_data in data.get('messages', []):
                    self.conversation_history.append(ConversationMessage(**msg_data))

                logger.info(f"✅ Loaded {len(self.conversation_history)} historical messages")
            else:
                logger.info("📝 No previous conversation history found (starting fresh)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load conversation history: {str(e)}")

    def _save_evaluation_metrics(self):
        """Save evaluation metrics to file."""
        try:
            metrics_data = {
                "conversation_id": self.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "evaluations": [eval_result.to_dict() for eval_result in self.evaluation_results]
            }

            with open(EVALUATION_METRICS_FILE, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            logger.debug("✅ Evaluation metrics saved")
        except Exception as e:
            logger.error(f"❌ Failed to save evaluation metrics: {str(e)}")

    def show_evaluation_metrics(self):
        """Display evaluation metrics summary."""
        if not self.evaluation_results:
            print("\n📊 No evaluation metrics available yet.\n")
            return

        print("\n" + "="*80)
        print("📊 RAGAS EVALUATION METRICS")
        print("="*80)

        # Calculate averages
        avg_context_rel = sum(e.metrics.context_relevance for e in self.evaluation_results) / len(self.evaluation_results)
        avg_answer_rel = sum(e.metrics.answer_relevance for e in self.evaluation_results) / len(self.evaluation_results)
        avg_faithful = sum(e.metrics.faithfulness for e in self.evaluation_results) / len(self.evaluation_results)
        avg_rag_score = sum(e.metrics.rag_score for e in self.evaluation_results) / len(self.evaluation_results)

        print(f"\n📈 OVERALL METRICS (from {len(self.evaluation_results)} evaluations):")
        print(f"  Context Relevance:  {avg_context_rel:.1%} ⭐")
        print(f"  Answer Relevance:   {avg_answer_rel:.1%} ⭐")
        print(f"  Faithfulness:       {avg_faithful:.1%} ⭐")
        print(f"  ─────────────────────────────")
        print(f"  Average RAG Score:  {avg_rag_score:.1%} 🎯")

        print(f"\n📚 RETRIEVAL METHODS:")
        methods = {}
        for result in self.evaluation_results:
            methods[result.retrieval_method] = methods.get(result.retrieval_method, 0) + 1
        for method, count in methods.items():
            print(f"  {method.upper()}: {count} times")

        print(f"\n🔍 RECENT EVALUATIONS:")
        table_data = []
        for eval_result in self.evaluation_results[-5:]:  # Last 5
            table_data.append([
                eval_result.query[:30] + "..." if len(eval_result.query) > 30 else eval_result.query,
                eval_result.retrieval_method,
                f"{eval_result.metrics.rag_score:.1%}",
                eval_result.num_chunks_retrieved
            ])

        print(tabulate(table_data, headers=["Query", "Method", "RAG Score", "Chunks"], tablefmt="grid"))
        print("="*80 + "\n")

    def print_response(self, response: RAGResponse, metrics: Optional[RAGASMetrics] = None):
        """Pretty print the RAG response with evaluation metrics."""
        print("\n" + "="*80)
        print("💡 ANSWER")
        print("="*80)
        print(response.answer)

        if response.sources:
            print("\n" + "="*80)
            print(f"📚 SOURCES & CONTEXT ({len(response.sources)} chunks retrieved)")
            print("="*80)

            for i, doc in enumerate(response.sources, 1):
                source_emoji = "🌐" if doc.source_type == "url" else "📚" if doc.source_type == "wikipedia" else "📄"
                print(f"\n[{i}] {source_emoji} {doc.source_type.upper()}")
                print(f"    Source: {doc.source}")
                print(f"    Relevance Score: {doc.relevance_score:.1%}")
                print(f"    Content Preview: {doc.content[:100]}...")

        print("\n" + "="*80)
        print("📊 METADATA")
        print("="*80)
        print(f"  Confidence Score: {response.confidence_score:.1%}")
        print(f"  Source Types Used: {', '.join(response.source_types).upper() if response.source_types else 'None'}")
        print(f"  Conversation ID: {self.conversation_id}")
        print(f"  Total Messages in History: {len(self.conversation_history)}")

        # Display RAGAS metrics if available (NEW)
        if metrics:
            print("\n" + "="*80)
            print("🎯 RAGAS EVALUATION METRICS")
            print("="*80)
            print(metrics)

        print("="*80 + "\n")

    def show_conversation_history(self):
        """Display full conversation history."""
        if not self.conversation_history:
            print("\n📝 No conversation history yet.\n")
            return

        print("\n" + "="*80)
        print("📜 CONVERSATION HISTORY")
        print("="*80)

        for i, msg in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if msg.role == "user" else "🤖"
            print(f"\n[{i}] {role_emoji} {msg.role.upper()} ({msg.timestamp})")

            if msg.confidence_score is not None:
                print(f"    Confidence: {msg.confidence_score:.1%}")

            print(f"    Message: {msg.content[:200]}...")

            if msg.sources:
                sources_info = ', '.join([f"{s.get('type', 'unknown')} ({s.get('source', 'unknown')[:30]})" for s in msg.sources])
                print(f"    Sources: {sources_info}")

        print("\n" + "="*80 + "\n")

    def show_loaded_sources(self):
        """Display all loaded sources."""
        if not self.loaded_sources:
            print("\n❌ No sources loaded yet.\n")
            return

        print("\n" + "="*80)
        print("📂 LOADED SOURCES")
        print("="*80)

        for source, source_type in self.loaded_sources.items():
            source_emoji = "🌐" if source_type == "url" else "📚" if source_type == "wikipedia" else "📄"
            print(f"{source_emoji} [{source_type.upper()}] {source}")

        print("="*80 + "\n")


def main():
    """Main interactive loop."""
    print("\n" + "="*80)
    print("🚀 Advanced RAG System: Hybrid Search + RAGAS Evaluation")
    print("="*80)
    print("\n✨ NEW FEATURES:")
    print("  ✓ Hybrid Search (BM25 + Semantic)")
    print("  ✓ RAGAS Evaluation Metrics")
    print("  ✓ Adaptive Chunk Sizing")
    print("  ✓ Multi-Source Support (Wikipedia, URLs, Local Files)")
    print("  ✓ Conversation History with Context Awareness")
    print("  ✓ Source Citation & Transparency")
    print("\n📋 Commands:")
    print("  'load <source>'  - Load Wikipedia page, URL, or file")
    print("  'sources'        - Show all loaded sources")
    print("  'history'        - Show conversation history")
    print("  'metrics'        - Show RAGAS evaluation metrics")
    print("  'clear'          - Clear conversation history")
    print("  'quit'           - Exit application")
    print("="*80 + "\n")

    rag_system = EnhancedRAGSystem()

    while True:
        try:
            user_input = input("❓ Enter command or ask a question: ").strip()

            if not user_input:
                print("⚠️ Please enter a valid input.")
                continue

            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break

            elif user_input.lower() == 'history':
                rag_system.show_conversation_history()

            elif user_input.lower() == 'metrics':
                rag_system.show_evaluation_metrics()

            elif user_input.lower() == 'sources':
                rag_system.show_loaded_sources()

            elif user_input.lower() == 'clear':
                rag_system.conversation_history = []
                rag_system._save_conversation_history()
                print("✅ Conversation history cleared.")

            elif user_input.lower().startswith('load '):
                source = user_input[5:].strip()
                if source:
                    rag_system.load_source(source)
                else:
                    print("❌ Please provide a source (Wikipedia page, URL, or file path).")

            else:
                # Check if any sources are loaded
                if not rag_system.loaded_sources:
                    print("⚠️ Please load at least one source first using: load <source>")
                    print("   Examples:")
                    print("   - load Albert Einstein")
                    print("   - load https://example.com/article")
                    print("   - load documents/article.txt")
                    continue

                # Process query through RAG pipeline with evaluation
                response, metrics = rag_system.process_query(user_input, enable_evaluation=True)
                rag_system.print_response(response, metrics)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
