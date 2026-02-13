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

# Load environment variables
load_dotenv()

# Configuration
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "lm-studio")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
MAX_TOOL_CALLS = 10
MAX_RETRIEVED_CHUNKS = 3
CONVERSATION_HISTORY_FILE = "./conversation_history.json"

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




db_client = chromadb.PersistentClient(path="./chroma_db")
db_client.heartbeat()
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Set a proper user agent for Wikipedia API
USER_AGENT = "generative-ai-learning/1.0 (contact: your-email@example.com)"
wiki = Wikipedia(user_agent=USER_AGENT, language="en")


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
        self._load_conversation_history()
        logger.info(f"✅ Initialized RAG System with conversation ID: {self.conversation_id}")

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
            # Split into paragraphs/chunks
            chunks = content.split("\n\n")
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

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

            self.collections[collection_name] = collection
            self.loaded_sources[source] = source_type

            print(f"✅ Successfully loaded {len(chunks)} chunks from {source}")
            print(f"   Source Type: {source_type.upper()}")
            print(f"   Collection: {collection_name}\n")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to store in ChromaDB: {str(e)}")
            print(f"❌ Error: {str(e)}")
            return False

    def _retrieve_relevant_chunks(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS) -> List[RetrievedDocument]:
        """Retrieve relevant chunks from all loaded collections."""
        logger.info(f"🔍 Retrieving chunks for query: {query[:50]}...")

        all_results = []

        for collection_name, collection in self.collections.items():
            try:
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

        logger.info(f"✅ Retrieved {len(top_results)} relevant chunks")
        return top_results

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

    def process_query(self, user_query: str) -> RAGResponse:
        """Process query through the RAG pipeline."""
        logger.info("=" * 80)
        logger.info(f"🚀 Processing query: {user_query[:60]}...")

        try:
            # Retrieve relevant chunks
            retrieved_docs = self._retrieve_relevant_chunks(user_query)

            # Generate answer
            answer, confidence = self._generate_answer(user_query, retrieved_docs)

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

            logger.info(f"✅ Query processed successfully")
            logger.info("=" * 80)

            return RAGResponse(
                answer=answer,
                sources=retrieved_docs,
                confidence_score=confidence,
                source_types=source_types,
                conversation_context=self._build_conversation_context()
            )

        except Exception as e:
            logger.error(f"❌ Query processing failed: {str(e)}")
            return RAGResponse(
                answer=f"Error: {str(e)}",
                sources=[],
                confidence_score=0.0,
                source_types=[],
                conversation_context=""
            )

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

    def print_response(self, response: RAGResponse, show_full_context: bool = True):
        """Pretty print the RAG response."""
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
    print("🚀 Enhanced RAG System: Multi-Source Q&A with Conversation History")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Multi-Source Support (Wikipedia, URLs, Local Files)")
    print("  ✓ Conversation History with Context Awareness")
    print("  ✓ Source Citation & Transparency")
    print("  ✓ Confidence Scores & Relevance Metrics")
    print("  ✓ Web Scraping Capabilities")
    print("\nCommands:")
    print("  'load <source>'  - Load Wikipedia page, URL, or file")
    print("  'sources'        - Show all loaded sources")
    print("  'history'        - Show conversation history")
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

                # Process query through RAG pipeline
                response = rag_system.process_query(user_input)
                rag_system.print_response(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
