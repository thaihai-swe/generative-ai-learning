# Enhanced RAG System: Multi-Source Q&A with Conversation History

A production-grade Retrieval-Augmented Generation (RAG) system that demonstrates advanced AI engineering practices. Features multi-source support, conversation memory, source attribution, and confidence scoring.

## ✨ Key Features

### 1. **Multi-Source Support**
- 📚 **Wikipedia Pages** - Load any Wikipedia article automatically
- 🌐 **Web URLs** - Scrape and index content from web pages
- 📄 **Local Files** - Process text, markdown, and documents
- 🔄 **Automatic Source Detection** - Intelligently detects source type

### 2. **Conversation History & Context**
- 💾 **Persistent Memory** - Saves conversations to JSON for future sessions
- 🔗 **Context-Aware Responses** - Considers previous messages when answering
- 📝 **Follow-up Support** - References earlier context automatically
- ⏱️ **Timestamped Messages** - Full audit trail of interactions

### 3. **Source Citation & Transparency**
- 🎯 **Source Attribution** - Shows exactly which document powered each answer
- 📊 **Relevance Scores** - Displays vector similarity confidence (0-100%)
- 🔍 **Chunk Preview** - Shows the relevant text passages used
- ✅ **Confidence Metrics** - Average score across all retrieved sources

### 4. **Advanced RAG Pipeline**
- 🔍 **Semantic Search** - Uses embeddings to find relevant content
- 🤝 **Multi-Collection Queries** - Searches across all loaded sources simultaneously
- 📈 **Smart Ranking** - Ranks results by relevance score
- 🧠 **Intelligent Answering** - LLM generates accurate answers based on context

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Setup

1. Create a `.env` file in the project directory:

```env
OPEN_AI_API_KEY=your_api_key
OPEN_AI_API_BASE_URL=http://127.0.0.1:1234/v1
OPEN_AI_MODEL=meta-llama-3.1-8b-instruct
```

2. Ensure ChromaDB directory exists:
```bash
mkdir -p chroma_db
```

### Running the System

```bash
python rag-chromadb.py
```

## 📖 Usage Examples

### Load Sources

```
❓ Enter command or ask a question: load Albert Einstein
✅ Successfully loaded 42 chunks from Albert Einstein
   Source Type: WIKIPEDIA
   Collection: albert_einstein

❓ Enter command or ask a question: load https://en.wikipedia.org/wiki/Machine_Learning
✅ Successfully loaded 156 chunks from https://en.wikipedia.org/wiki/Machine_Learning
   Source Type: URL
   Collection: en_wikipedia_org

❓ Enter command or ask a question: load documents/research_paper.txt
✅ Successfully loaded 89 chunks from documents/research_paper.txt
   Source Type: FILE
   Collection: documents_research_paper
```

### Ask Questions

```
❓ Enter command or ask a question: What are Einstein's major contributions?

💡 ANSWER
===============================================================================
Einstein's major contributions to physics include:

1. Theory of Special Relativity (1905) - Revolutionized understanding of space and time
2. Theory of General Relativity (1915) - Explained gravity as curvature of spacetime
3. Photoelectric Effect - Explained light as quanta, earning him the Nobel Prize

[Source 1 - WIKIPEDIA]
...

📚 SOURCES & CONTEXT (3 chunks retrieved)
===============================================================================
[1] 🌐 WIKIPEDIA
    Source: Albert Einstein
    Relevance Score: 95.3%
    Content Preview: Einstein was a German-born theoretical physicist...

📊 METADATA
===============================================================================
  Confidence Score: 94.2%
  Source Types Used: WIKIPEDIA
  Conversation ID: 20260213_142530
  Total Messages in History: 2
===============================================================================
```

### View Conversation History

```
❓ Enter command or ask a question: history

📜 CONVERSATION HISTORY
===============================================================================
[1] 👤 USER (2026-02-13T14:25:30.123456)
    Message: What are Einstein's major contributions?
    Sources: wikipedia (Albert Einstein), wikipedia (Albert Einstein)

[2] 🤖 ASSISTANT (2026-02-13T14:25:35.456789)
    Confidence: 94.2%
    Message: Einstein's major contributions to physics include...
    Sources: wikipedia (Albert Einstein)

[3] 👤 USER (2026-02-13T14:26:10.789012)
    Message: How did these theories change physics?
    Sources: wikipedia (Albert Einstein)
===============================================================================
```

### Available Commands

| Command         | Description                       |
| --------------- | --------------------------------- |
| `load <source>` | Load Wikipedia page, URL, or file |
| `sources`       | Show all loaded sources           |
| `history`       | Display conversation history      |
| `clear`         | Clear all conversation history    |
| `quit`          | Exit application                  |

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│              Enhanced RAG System                         │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  MultiSourceDataLoader                            │  │
│  │  - scrape_url()     [BeautifulSoup]               │  │
│  │  - load_wikipedia() [Wikipedia API]               │  │
│  │  - load_file()      [File I/O]                    │  │
│  │  - detect_source_type()                           │  │
│  └────────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  ChromaDB Vector Store                            │  │
│  │  - Multiple Collections (one per source)          │  │
│  │  - Default Embedding Function                     │  │
│  │  - Semantic Search                                │  │
│  └────────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  RAG Pipeline                                     │  │
│  │  1. Retrieve Documents   [Semantic Search]        │  │
│  │  2. Calculate Relevance  [Distance → Score]       │  │
│  │  3. Build Context        [Concatenate Chunks]     │  │
│  │  4. Generate Answer      [LLM + Prompting]        │  │
│  │  5. Track Sources        [Metadata]               │  │
│  └────────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Conversation Memory                              │  │
│  │  - Store in JSON         [Persistence]            │  │
│  │  - Load on Startup       [Memory Recovery]        │  │
│  │  - Context Awareness     [Follow-ups]             │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Data Models

```python
RetrievedDocument
  ├── content: str
  ├── source: str
  ├── source_type: str (wikipedia|url|file)
  ├── index: int
  ├── distance: Optional[float]
  └── relevance_score: Property[0.0-1.0]

ConversationMessage
  ├── role: str (user|assistant)
  ├── content: str
  ├── timestamp: str (ISO 8601)
  ├── sources: List[Dict]
  └── confidence_score: Optional[float]

RAGResponse
  ├── answer: str
  ├── sources: List[RetrievedDocument]
  ├── confidence_score: float
  ├── source_types: List[str]
  └── conversation_context: str
```

## 💡 How It Works

### 1. Loading a Source

```
User Input: "load Machine Learning"
    ↓
detect_source_type("Machine Learning") → "wikipedia"
    ↓
wiki.page("Machine Learning").text → "Machine learning is..."
    ↓
split("\n\n") → [chunk1, chunk2, chunk3, ...]
    ↓
collection.add(ids, documents, metadatas)
    ↓
✅ Stored in ChromaDB with embeddings
```

### 2. Processing a Query

```
User Question: "What is supervised learning?"
    ↓
retrieve_relevant_chunks(query) → [doc1, doc2, doc3]
    ↓
calculate relevance_scores() → [0.95, 0.87, 0.76]
    ↓
build_conversation_context() → "Previous: ...\n"
    ↓
LLM generates answer with context + sources
    ↓
store in conversation_history.json
    ↓
display answer + sources + confidence
```

## 📊 Response Structure

Each response includes:

1. **Answer** - The generated answer based on context
2. **Sources** - List of documents used:
   - Source link/name
   - Source type icon (🌐/📚/📄)
   - Relevance score (%)
   - Content preview
3. **Metadata** - System information:
   - Overall confidence score
   - Source types used
   - Conversation ID
   - Message count

## 🔧 Configuration

Edit these constants in `rag-chromadb.py`:

```python
MAX_RETRIEVED_CHUNKS = 3  # Results per query
CONVERSATION_HISTORY_FILE = "./conversation_history.json"
```

Edit `.env`:

```env
OPEN_AI_API_KEY=your_key
OPEN_AI_API_BASE_URL=http://127.0.0.1:1234/v1
OPEN_AI_MODEL=meta-llama-3.1-8b-instruct
```

## 📝 Conversation History

History is automatically saved to `conversation_history.json`:

```json
{
  "conversation_id": "20260213_142530",
  "timestamp": "2026-02-13T14:25:30.000000",
  "messages": [
    {
      "role": "user",
      "content": "What is Einstein known for?",
      "timestamp": "2026-02-13T14:25:30.000000",
      "sources": [
        {
          "source": "Albert Einstein",
          "type": "wikipedia"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "Einstein is known for developing the theories of...",
      "timestamp": "2026-02-13T14:25:35.000000",
      "confidence_score": 0.942,
      "sources": [
        {
          "source": "Albert Einstein",
          "type": "wikipedia"
        }
      ]
    }
  ]
}
```

## 🎯 Portfolio Highlights

This project demonstrates:

✅ **RAG Implementation** - Complete multi-source RAG pipeline
✅ **Multi-Source Integration** - Wikipedia, web scraping, file loading
✅ **Conversation Memory** - Persistent state management
✅ **Source Attribution** - Transparency and trustworthiness
✅ **Confidence Scoring** - Quality metrics for results
✅ **Error Handling** - Robust exception management
✅ **Structured Logging** - Production-level monitoring
✅ **Data Modeling** - Type-safe Python with dataclasses
✅ **API Integration** - OpenAI/Local LLM compatibility
✅ **User Experience** - Interactive CLI with helpful commands

## 🚦 Requirements

- Python 3.8+
- ChromaDB 0.4+
- OpenAI SDK 1.0+
- BeautifulSoup4 4.12+
- Requests 2.31+
- wikipediaapi 0.6+

## 📦 Dependencies

All dependencies are listed in `requirements.txt`

## 🐛 Troubleshooting

### "No sources loaded" error
→ Load a source first: `load Albert Einstein`

### Web scraping fails
→ Check internet connection, URL is valid, and server isn't blocking requests

### ChromaDB errors
→ Ensure `chroma_db/` directory exists and is writable

### Memory issues with large files
→ Reduce chunk size or split large files into smaller ones

## 🎓 Learning Resources

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Variable-Length Context Windows in LLMs](https://www.anthropic.com/news/100k-context-windows)

## 📈 Future Enhancements

- [ ] RAGAS evaluation metrics
- [ ] Hybrid search (BM25 + semantic)
- [ ] Query expansion
- [ ] Re-ranking with cross-encoders
- [ ] Multi-hop reasoning
- [ ] Web UI with FastAPI
- [ ] PostgreSQL + pgvector upgrade
- [ ] Redis caching layer
- [ ] Cost tracking dashboard

## 📜 License

MIT

## 💬 Contributing

Suggestions and improvements welcome! This is a portfolio project showcasing AI engineering practices.
