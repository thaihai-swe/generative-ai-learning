"""Core RAG orchestrator - coordinates all components"""
from typing import List, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from urllib.parse import urlparse

from src.config import RAGConfig, get_config
from src.models import RetrievedDocument, RAGResponse, RAGASMetrics
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.chunker import AdaptiveChunker
from src.retrieval.loader import MultiSourceDataLoader
from src.retrieval.cache import EmbeddingCache
from src.generation import LLMAnswerGenerator
from src.evaluation import RAGASEvaluator, FactChecker
from src.reasoning import QueryExpander, MultiHopReasoner
from src.persistence import JSONStorage
from src.utils import get_logger

logger = get_logger()


class RAGSystem:
    """Main RAG orchestrator - coordinates all components"""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_config()

        # Initialize components
        self.retriever = HybridSearchEngine(
            semantic_weight=self.config.search.semantic_weight,
            keyword_weight=self.config.search.keyword_weight
        )
        self.chunker = AdaptiveChunker()
        self.loader = MultiSourceDataLoader()
        self.embedding_cache = EmbeddingCache(max_size=self.config.search.embedding_cache_size)
        self.generator = LLMAnswerGenerator()
        self.evaluator = RAGASEvaluator()
        self.fact_checker = FactChecker()
        self.query_expander = QueryExpander()
        self.multi_hop_reasoner = MultiHopReasoner()
        self.storage = JSONStorage(self.config.data_dir)

        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(
            path=self.config.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.db.heartbeat()
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # State
        self.conversation_history = []
        self.current_collection = None
        self.collections = {}
        self.loaded_sources = {}
        self.last_fact_check_results = []

        logger.info("✅ Enhanced RAG System initialized")

    def _get_collection_name(self, source: str) -> str:
        """Generate sanitized collection name from source"""
        import re

        # Remove quotes if present
        source = source.strip('"\'')

        # Parse URL if it is one
        parsed = urlparse(source)
        if parsed.netloc:
            name = parsed.netloc.replace('.', '_').replace('/', '_')
        else:
            name = source

        # Replace spaces and special chars with underscores
        name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Remove any invalid characters (keep only alphanumeric, underscores, hyphens)
        name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)

        # Remove consecutive underscores/hyphens
        name = re.sub(r'_+', '_', name)
        name = re.sub(r'-+', '-', name)

        # Convert to lowercase
        name = name.lower()

        # Ensure length is 3-63 characters
        if len(name) < 3:
            name = f"col_{name}"  # Add prefix if too short
        elif len(name) > 63:
            name = name[:63]

        # Ensure starts and ends with alphanumeric
        name = re.sub(r'^[^a-z0-9]+', '', name)  # Remove leading non-alphanumeric
        name = re.sub(r'[^a-z0-9]+$', '', name)  # Remove trailing non-alphanumeric

        return name

    def load_data(self, source: str, collection_name: str = None) -> None:
        """Load and process data from source"""
        logger.info(f"📥 Loading data from {source}...")

        if collection_name is None:
            collection_name = self._get_collection_name(source)

        try:
            # Load content
            content = self.loader.load(source)
            if not content:
                print(f"❌ Failed to load from {source}")
                return

            # Detect source type
            if source.startswith(('http://', 'https://')):
                source_type = 'url'
            elif source.endswith(('.txt', '.md', '.pdf')):
                source_type = 'file'
            else:
                source_type = 'wikipedia'

            # Chunk content
            chunks = self.chunker.chunk(content)
            if not chunks:
                print(f"❌ No content to chunk from {source}")
                return

            logger.info(f"✅ Split into {len(chunks)} chunks")

            # Create ChromaDB collection
            collection = self.db.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

            # Add to collection
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    ids=[f"{collection_name}_{i}"],
                    metadatas=[{
                        "source": source,
                        "source_type": source_type,
                        "index": i,
                        "timestamp": datetime.now().isoformat()
                    }]
                )

            # Build BM25 index for hybrid search
            self.retriever.build_bm25_index(collection_name, chunks)
            logger.info(f"✅ Built BM25 index for {collection_name}")

            self.collections[collection_name] = collection
            self.current_collection = collection
            self.loaded_sources[source] = source_type

            print(f"✅ Successfully loaded {len(chunks)} chunks from {source}")
            print(f"   Source Type: {source_type.upper()}")
            print(f"   Collection: {collection_name}\n")

            logger.info(f"✅ Data loaded into collection: {collection_name}")

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            print(f"❌ Error: {e}")

    def process_query(self, query: str, use_expansion: bool = False) -> RAGResponse:
        """End-to-end query processing"""
        logger.info(f"❓ Processing query: {query}")

        start_time = datetime.now()

        try:
            # 1. Query expansion (only if explicitly requested)
            queries = (
                self.query_expander.expand(query, 4)  # Always 4 variations when expansion is used
                if use_expansion
                else [query]
            )

            # 2. Retrieve documents
            docs = self._retrieve_documents(queries[0])  # Use primary query for retrieval

            if not docs:
                return RAGResponse(
                    answer="No relevant documents found.",
                    sources=[],
                    confidence_score=0.0,
                    source_types=[],
                    conversation_context=None,
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )

            # 3. Generate answer
            answer = self.generator.generate(query, docs)

            # 4. Evaluate (if enabled)
            confidence_score = 0.5
            if self.config.evaluation.compute_ragas:
                context = "\n".join([doc.content for doc in docs])
                metrics = self.evaluator.evaluate(query, context, answer)
                confidence_score = metrics.rag_score

            # 5. Fact-check (if enabled)
            fact_results = None
            if self.config.evaluation.check_facts and docs:
                context = "\n".join([doc.content for doc in docs])
                fact_results = self.fact_checker.check_answer(answer, context)
                self.last_fact_check_results = fact_results  # Store for CLI access
                logger.info(f"✅ Fact-checked {len(fact_results)} claims")

            # 6. Store in history
            self.conversation_history.append({
                "query": query,
                "answer": answer,
                "sources": [doc.source for doc in docs],
                "timestamp": datetime.now().isoformat()
            })

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            response = RAGResponse(
                answer=answer,
                sources=docs,
                confidence_score=confidence_score,
                source_types=list(set(doc.source_type for doc in docs)),
                conversation_context=None,
                execution_time_ms=execution_time,
                fact_check_results=fact_results
            )

            logger.info(f"✅ Query processed in {execution_time:.1f}ms")
            return response

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return RAGResponse(
                answer=f"Error: {str(e)}",
                sources=[],
                confidence_score=0.0,
                source_types=[],
                conversation_context=None,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _retrieve_documents(self, query: str) -> List[RetrievedDocument]:
        """Retrieve relevant documents using hybrid search"""
        if not self.current_collection:
            logger.warning("⚠️ No collection loaded")
            return []

        try:
            collection_name = None
            for name, col in self.collections.items():
                if col == self.current_collection:
                    collection_name = name
                    break

            if not collection_name:
                logger.warning("⚠️ Could not find collection name")
                return []

            # 1. Semantic search via ChromaDB
            semantic_results = self.current_collection.query(
                query_texts=[query],
                n_results=self.config.search.max_results
            )

            # Build semantic results tuples (content, distance)
            semantic_docs = []
            for i, doc in enumerate(semantic_results['documents'][0]):
                distance = semantic_results['distances'][0][i] if semantic_results['distances'] else 1.0
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = max(0, 1 - (distance / 2))
                semantic_docs.append((doc, similarity))

            # 2. Keyword search via BM25
            keyword_docs = self.retriever.keyword_search(
                collection_name,
                query,
                top_k=self.config.search.max_results
            )

            # 3. Combine using hybrid search
            if semantic_docs and keyword_docs:
                combined = self.retriever._combine_results(semantic_docs, keyword_docs)
                logger.info(f"🔍 Hybrid search: {len(semantic_docs)} semantic + {len(keyword_docs)} keyword → {len(combined)} combined")
            elif semantic_docs:
                combined = [(doc, score) for doc, score in semantic_docs]
                logger.info(f"🔍 Semantic-only search: {len(semantic_docs)} results")
            elif keyword_docs:
                combined = [(doc, score) for doc, score in keyword_docs]
                logger.info(f"🔍 Keyword-only search: {len(keyword_docs)} results")
            else:
                logger.warning("⚠️ No results from either search method")
                return []

            # 4. Convert to RetrievedDocument objects
            documents = []
            for content, score in combined[:self.config.search.max_results]:
                # Find metadata from original semantic results
                idx = -1
                for i, doc in enumerate(semantic_results['documents'][0]):
                    if doc == content:
                        idx = i
                        break

                if idx >= 0:
                    source_type = semantic_results['metadatas'][0][idx].get('source_type', 'unknown')
                    source = semantic_results['metadatas'][0][idx].get('source', 'unknown')
                    index = semantic_results['metadatas'][0][idx].get('index', idx)
                else:
                    source_type = 'unknown'
                    source = 'unknown'
                    index = len(documents)

                retrieved_doc = RetrievedDocument(
                    content=content,
                    source=source,
                    source_type=source_type,
                    index=index,
                    distance=1.0 - score  # Convert score back to distance
                )
                documents.append(retrieved_doc)

            logger.info(f"✅ Retrieved {len(documents)} documents via hybrid search")
            return documents

        except Exception as e:
            logger.error(f"❌ Hybrid retrieval failed: {e}, falling back to semantic only")
            # Fallback to semantic-only search
            try:
                results = self.current_collection.query(
                    query_texts=[query],
                    n_results=self.config.search.max_results
                )

                documents = []
                for i, doc in enumerate(results['documents'][0]):
                    source_type = results['metadatas'][0][i].get('source_type', 'collection')
                    retrieved_doc = RetrievedDocument(
                        content=doc,
                        source=results['metadatas'][0][i].get('source', 'unknown'),
                        source_type=source_type,
                        index=results['metadatas'][0][i].get('index', i),
                        distance=results['distances'][0][i] if results['distances'] else None
                    )
                    documents.append(retrieved_doc)

                logger.info(f"✅ Retrieved {len(documents)} documents (fallback)")
                return documents

            except Exception as fallback_error:
                logger.error(f"❌ Fallback retrieval failed: {fallback_error}")
                return []

    def _retrieve_relevant_chunks(self, query: str, n_results: int = 3) -> List[RetrievedDocument]:
        """Retrieve relevant chunks using semantic search"""
        all_results = []

        for collection_name, collection in self.collections.items():
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results
                )

                for i, doc in enumerate(results['documents'][0]):
                    source_type = results['metadatas'][0][i].get('source_type', 'unknown')
                    retrieved_doc = RetrievedDocument(
                        content=doc,
                        source=results['metadatas'][0][i].get('source', 'unknown'),
                        source_type=source_type,
                        index=results['metadatas'][0][i].get('index', i),
                        distance=results['distances'][0][i] if results['distances'] else None
                    )
                    all_results.append(retrieved_doc)
            except Exception as e:
                logger.warning(f"⚠️ Query failed for {collection_name}: {e}")

        all_results.sort(key=lambda x: x.distance if x.distance else float('inf'))
        return all_results[:n_results]

    def save_conversation(self, filename: str = "conversation") -> None:
        """Save conversation history"""
        self.storage.save(self.conversation_history, filename)
        logger.info(f"💾 Conversation saved to {filename}")

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = self.embedding_cache.get_stats()
        return {
            "cache_size": stats.get('size', 0),
            "cache_hit_rate": stats.get('hit_rate', '0%'),
            "total_queries": stats.get('total_lookups', 0)
        }


__all__ = ["RAGSystem"]
