import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pprint import pprint
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from wikipediaapi import Wikipedia

# Load environment variables
load_dotenv()

# Configuration
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "lm-studio")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
MAX_TOOL_CALLS = 10
MAX_RETRIEVED_CHUNKS = 3

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
    sources: Optional[List[str]] = None


@dataclass
class RAGResponse:
    """Structured RAG response with metadata."""
    answer: str
    sources: List[RetrievedDocument]
    confidence_score: float
    retrieval_method: str
    conversation_id: str




db_client = chromadb.PersistentClient(path="./chroma_db")
db_client.heartbeat()
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Set a proper user agent for Wikipedia API
USER_AGENT = "generative-ai-learning/1.0 (contact: your-email@example.com)"
wiki = Wikipedia(user_agent=USER_AGENT, language="en")


# Function Calling Definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search and retrieve content from Wikipedia for a given topic. Use this when you need external information to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_title": {
                        "type": "string",
                        "description": "The Wikipedia page title to search for (e.g., 'Albert Einstein', 'Machine Learning')"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this information is needed to answer the user's question"
                    }
                },
                "required": ["page_title", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_local_knowledge",
            "description": "Query the local ChromaDB vector database for information already stored. Use this for faster retrieval of previously indexed content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query text to search in the vector database"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to retrieve (1-5)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class RAGOrchestrator:
    """Main orchestrator for the RAG pipeline with Function Calling."""

    def __init__(self):
        self.current_collection = None
        self.current_page = None
        self.conversation_history: List[ConversationMessage] = []
        self.conversation_id = self._generate_conversation_id()
        logger.info(f"Initialized RAG Orchestrator with conversation ID: {self.conversation_id}")

    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _detect_intent(self, user_query: str) -> Dict:
        """
        Step 1: Intent Detection using Function Calling.
        Determines if external information is needed and which tools to use.
        """
        logger.info(f"🎯 Intent Detection: Analyzing query - {user_query[:50]}...")

        try:
            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at understanding user queries and determining if external information is needed.
                        Analyze the user's question and decide:
                        1. If you need to search Wikipedia for new information
                        2. If you can query locally stored knowledge
                        3. If you can answer without external data

                        Use function calling strategically to gather information."""
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3
            )

            intent_result = {
                "requires_search": False,
                "tools_to_use": [],
                "reasoning": response.choices[0].message.content
            }

            # Check if function calling was triggered
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    intent_result["requires_search"] = True
                    intent_result["tools_to_use"].append({
                        "name": tool_call.function.name,
                        "args": json.loads(tool_call.function.arguments),
                        "call_id": tool_call.id
                    })

            logger.info(f"✅ Intent detected - Requires search: {intent_result['requires_search']}")
            return intent_result

        except Exception as e:
            logger.error(f"❌ Intent detection failed: {str(e)}")
            return {"requires_search": False, "tools_to_use": [], "reasoning": "Error in intent detection"}

    def _retrieve_wikipedia_data(self, page_title: str) -> Optional[str]:
        """
        Step 2: Data Retrieval from Wikipedia API.
        Fetches and processes content from the specified Wikipedia page.
        """
        logger.info(f"📚 Data Retrieval: Fetching Wikipedia page - {page_title}")

        try:
            page = wiki.page(page_title)

            if not page.exists():
                logger.warning(f"⚠️ Wikipedia page '{page_title}' not found")
                return None

            text = page.text
            if not text:
                logger.warning(f"⚠️ No content retrieved from '{page_title}'")
                return None

            logger.info(f"✅ Retrieved {len(text)} characters from '{page_title}'")
            return text

        except Exception as e:
            logger.error(f"❌ Wikipedia retrieval failed: {str(e)}")
            return None

    def _store_in_vectordb(self, content: str, page_name: str) -> Optional[chromadb.Collection]:
        """
        Step 3: Vector Storage in ChromaDB.
        Processes retrieved text and stores embeddings.
        """
        logger.info(f"💾 Vector Storage: Processing and storing embeddings for {page_name}")

        try:
            docs = content.split("\n\n")  # Split into paragraphs
            docs = [doc.strip() for doc in docs if doc.strip()]  # Remove empty docs

            if not docs:
                logger.warning(f"⚠️ No valid documents to store for {page_name}")
                return None

            # Sanitize collection name for ChromaDB
            collection_name = page_name.replace(" ", "_").replace("/", "_").lower()

            collection = db_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )

            # Add documents with metadata
            collection.add(
                ids=[f"{collection_name}_{i}" for i in range(len(docs))],
                documents=docs,
                metadatas=[
                    {
                        "source": page_name,
                        "index": i,
                        "timestamp": datetime.now().isoformat()
                    } for i in range(len(docs))
                ]
            )

            self.current_collection = collection
            self.current_page = page_name

            logger.info(f"✅ Stored {len(docs)} document chunks in ChromaDB")
            return collection

        except Exception as e:
            logger.error(f"❌ Vector storage failed: {str(e)}")
            return None

    def _retrieve_relevant_chunks(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS) -> List[RetrievedDocument]:
        """
        Step 4.1: Retrieve relevant chunks from ChromaDB using semantic search.
        """
        logger.info(f"🔍 Semantic Search: Querying for top-{n_results} relevant chunks")

        if not self.current_collection:
            logger.warning("⚠️ No collection loaded. Cannot retrieve chunks.")
            return []

        try:
            results = self.current_collection.query(
                query_texts=[query],
                n_results=min(n_results, MAX_RETRIEVED_CHUNKS),
                include=["documents", "metadatas", "distances"]
            )

            if not results['documents'] or not results['documents'][0]:
                logger.warning("⚠️ No relevant documents found")
                return []

            retrieved_docs = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else None

                retrieved_docs.append(RetrievedDocument(
                    content=doc,
                    source=metadata.get('source', 'Unknown'),
                    index=metadata.get('index', 0),
                    distance=distance
                ))

            logger.info(f"✅ Retrieved {len(retrieved_docs)} relevant chunks")
            for i, doc in enumerate(retrieved_docs):
                logger.debug(f"  Chunk {i+1} (relevance: {doc.relevance_score:.2f}): {doc.content[:60]}...")

            return retrieved_docs

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {str(e)}")
            return []

    def _generate_answer(self, query: str, context_docs: List[RetrievedDocument]) -> Tuple[str, float]:
        """
        Step 4.2: Answer Generation using RAG pipeline.
        Generates answer based on retrieved context with confidence scoring.
        """
        logger.info(f"🤖 Answer Generation: Generating response with {len(context_docs)} context chunks")

        try:
            # Build context string from retrieved documents
            context = ""
            for i, doc in enumerate(context_docs, 1):
                context += f"[Source {i}: {doc.source}]\n{doc.content}\n\n"

            if not context:
                logger.warning("⚠️ No context available for answer generation")
                return "I don't have enough information to answer your question.", 0.3

            # Calculate average confidence from retrieved documents
            avg_confidence = sum(doc.relevance_score for doc in context_docs) / len(context_docs) if context_docs else 0.5

            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that provides accurate, concise answers based on provided context.
                        - Only use information from the provided context
                        - If information is not in the context, say "Based on the provided context, I cannot answer this"
                        - Be precise and cite which source you're using when relevant
                        - Keep answers focused and well-structured"""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear, accurate answer based on the context."
                    }
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
        """
        Main RAG pipeline orchestration.
        Follows the workflow: Intent Detection → Data Retrieval → Vector Storage → Answer Generation
        """
        logger.info("=" * 80)
        logger.info(f"🚀 Starting RAG Pipeline for query: {user_query[:50]}...")

        try:
            # Step 1: Intent Detection
            intent = self._detect_intent(user_query)

            # Step 2 & 3: If external data is needed, retrieve from Wikipedia and store in ChromaDB
            if intent["requires_search"]:
                for tool in intent["tools_to_use"]:
                    if tool["name"] == "search_wikipedia":
                        page_title = tool["args"].get("page_title")
                        reason = tool["args"].get("reason")
                        logger.info(f"📖 Function Call Triggered: search_wikipedia")
                        logger.info(f"   Page: {page_title}")
                        logger.info(f"   Reason: {reason}")

                        # Retrieve content from Wikipedia
                        content = self._retrieve_wikipedia_data(page_title)

                        if content:
                            # Store in ChromaDB
                            self._store_in_vectordb(content, page_title)

            # Step 4: Retrieve relevant chunks from ChromaDB
            retrieved_docs = self._retrieve_relevant_chunks(user_query)

            # Step 5: Generate answer using RAG
            answer, confidence = self._generate_answer(user_query, retrieved_docs)

            # Store in conversation history
            source_refs = [f"{doc.source} (para {doc.index})" for doc in retrieved_docs]
            self.conversation_history.append(ConversationMessage(
                role="user",
                content=user_query,
                timestamp=datetime.now().isoformat(),
                sources=source_refs
            ))
            self.conversation_history.append(ConversationMessage(
                role="assistant",
                content=answer,
                timestamp=datetime.now().isoformat(),
                sources=source_refs
            ))

            logger.info(f"✅ RAG Pipeline completed successfully")
            logger.info("=" * 80)

            return RAGResponse(
                answer=answer,
                sources=retrieved_docs,
                confidence_score=confidence,
                retrieval_method="semantic_search" if retrieved_docs else "none",
                conversation_id=self.conversation_id
            )

        except Exception as e:
            logger.error(f"❌ RAG Pipeline failed: {str(e)}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                retrieval_method="error",
                conversation_id=self.conversation_id
            )

    def print_response(self, response: RAGResponse):
        """Pretty print the RAG response with all metadata."""
        print("\n" + "="*80)
        print(f"💡 ANSWER:")
        print("-"*80)
        print(response.answer)
        print("-"*80)

        if response.sources:
            print(f"\n📚 SOURCES ({len(response.sources)} chunks):")
            for i, doc in enumerate(response.sources, 1):
                print(f"\n  [{i}] {doc.source} (Paragraph {doc.index})")
                print(f"      Relevance: {doc.relevance_score:.1%}")
                print(f"      Content: {doc.content[:80]}...")

        print(f"\n📊 METADATA:")
        print(f"  Confidence Score: {response.confidence_score:.1%}")
        print(f"  Retrieval Method: {response.retrieval_method}")
        print(f"  Conversation ID: {response.conversation_id}")
        print("="*80 + "\n")

    def show_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("\n📝 No conversation history yet.\n")
            return

        print("\n" + "="*80)
        print("📜 CONVERSATION HISTORY")
        print("="*80)

        for i, msg in enumerate(self.conversation_history, 1):
            role = "👤 USER" if msg.role == "user" else "🤖 ASSISTANT"
            print(f"\n[{i}] {role} ({msg.timestamp})")
            print(f"    {msg.content[:150]}...")
            if msg.sources:
                print(f"    Sources: {', '.join(msg.sources)}")

        print("\n" + "="*80 + "\n")


def main():
    """Main interactive loop with enhanced RAG system."""
    print("\n" + "="*80)
    print("🚀 RAG-based Q&A System with Function Calling, Wikipedia & ChromaDB")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Intent Detection using Function Calling")
    print("  ✓ Dynamic Wikipedia Data Retrieval")
    print("  ✓ Vector Storage in ChromaDB")
    print("  ✓ Advanced RAG Pipeline with Confidence Scores")
    print("  ✓ Source Attribution & Conversation History")
    print("\nCommands:")
    print("  'history' - Show conversation history")
    print("  'clear'   - Clear conversation history")
    print("  'page'    - Load a new Wikipedia page")
    print("  'quit'    - Exit the application")
    print("="*80 + "\n")

    orchestrator = RAGOrchestrator()

    while True:
        print(f"\n📚 Current Page: {orchestrator.current_page or 'None'}")
        user_input = input("\n❓ Ask your question (or enter command): ").strip()

        if not user_input:
            print("⚠️ Please enter a valid input.")
            continue

        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break

        elif user_input.lower() == 'history':
            orchestrator.show_conversation_history()

        elif user_input.lower() == 'clear':
            orchestrator.conversation_history = []
            print("✅ Conversation history cleared.")

        elif user_input.lower() == 'page':
            page_name = input("📝 Enter Wikipedia page name: ").strip()
            if page_name:
                content = orchestrator._retrieve_wikipedia_data(page_name)
                if content:
                    orchestrator._store_in_vectordb(content, page_name)
                    print(f"✅ Successfully loaded '{page_name}'")
            continue

        else:
            # Process query through RAG pipeline
            response = orchestrator.process_query(user_input)
            orchestrator.print_response(response)


if __name__ == "__main__":
    main()
