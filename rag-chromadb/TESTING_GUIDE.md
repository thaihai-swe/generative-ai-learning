🚀 COMPLETE TESTING GUIDE - ALL COMMANDS WITH EXAMPLES

═══════════════════════════════════════════════════════════════════════════════

QUICK START:

1. Start the system:
   $ python3 launch.py

2. Then copy-paste the examples below one by one to test each feature

═══════════════════════════════════════════════════════════════════════════════

TEST PLAN (Run in order):
═══════════════════════════════════════════════════════════════════════════════

✅ TEST 1: Load Data from Wikipedia
─────────────────────────────────────────────────────────────────────────────

COMMAND:
  load wikipedia "Machine Learning"

EXPECTED OUTPUT:
  ✅ Successfully loaded 15+ chunks from Machine Learning
     Source Type: WIKIPEDIA
     Collection: wikipedia_machine_learning

HOW IT WORKS:
  • "wikipedia" = magic keyword that triggers Wikipedia loader
  • "Machine Learning" = page title to search for
  • System fetches the Wikipedia page and splits it into chunks
  • Each chunk is stored in ChromaDB for searching

WHAT TO TRY:
  load wikipedia "Albert Einstein"
  load wikipedia "Python Programming"
  load wikipedia "Artificial Intelligence"
  load wikipedia "Quantum Physics"
  load wikipedia "Cloud Computing"


✅ TEST 2: Load Data from URL
─────────────────────────────────────────────────────────────────────────────

COMMAND:
  load https://en.wikipedia.org/wiki/Machine_learning

EXPECTED OUTPUT:
  ✅ Successfully loaded X chunks from URL
     Source Type: URL
     Collection: en_wikipedia_org

HOW IT WORKS:
  • System detects HTTP/HTTPS URLs automatically
  • Scrapes HTML content using BeautifulSoup
  • Removes script/style elements
  • Chunks the text for storage

WHAT TO TRY:
  load https://www.wikipedia.org/wiki/Artificial_intelligence
  load https://example.com/article
  load https://docs.python.org/3/


✅ TEST 3: Load Data from Local File
─────────────────────────────────────────────────────────────────────────────

FIRST, CREATE A TEST FILE:

  Create a file called test_document.txt with content:

  ------- test_document.txt -------
  Machine Learning is a field of artificial intelligence that focuses
  on the development of algorithms and models that can learn from data
  without being explicitly programmed. It involves training models on
  historical data to make predictions or decisions on new, unseen data.

  Key concepts include supervised learning, unsupervised learning, and
  reinforcement learning. Common algorithms include decision trees,
  neural networks, and support vector machines.
  ---------------------------------

COMMAND TO LOAD IT:
  load test_document.txt

EXPECTED OUTPUT:
  ✅ Successfully loaded 2+ chunks from test_document.txt
     Source Type: FILE
     Collection: test_document

HOW IT WORKS:
  • System detects .txt, .md, .pdf file extensions
  • Reads local file content
  • Chunks and stores in database

WHAT TO TRY:
  load readme.txt
  load notes.md
  load mydata.txt


✅ TEST 4: Load Data from PDF
─────────────────────────────────────────────────────────────────────────────

FIRST, CREATE OR USE A PDF FILE

COMMAND TO LOAD IT:
  load mydocument.pdf

EXPECTED OUTPUT:
  ✅ Successfully loaded X chunks from mydocument.pdf
     Source Type: FILE
     Collection: mydocument

HOW IT WORKS:
  • System detects .pdf extension
  • Uses PyPDF2 to extract text from all pages
  • Combines text and chunks it
  • Stores in database


✅ TEST 5: Query the Loaded Data
─────────────────────────────────────────────────────────────────────────────

FIRST: Make sure you've loaded data with one of the above commands

COMMAND:
  query What is machine learning?

EXPECTED OUTPUT:
  💡 ANSWER:
  ════════════════════════════════
  [System responds with answer]
  ════════════════════════════════
  📊 Confidence: 75.3%
  📚 Sources: wikipedia
  📖 Retrieved 3 documents

HOW IT WORKS:
  • System searches loaded documents using hybrid search
  • Generates answer using LLM
  • Shows sources and confidence score

WHAT TO TRY:
  query What is artificial intelligence?
  query How does machine learning work?
  query What are neural networks?
  query Explain supervised learning


✅ TEST 6: Query Expansion
──────────────────────────────────────────────────────────────────────────

COMMAND:
  expand What is machine learning?

EXPECTED OUTPUT:
  🔄 Generating query variations...
  📝 Original: What is machine learning?
  🔄 Variations (4):
     1. What is machine learning?
     2. How does machine learning work?
     3. Explain machine learning principles
     4. What are applications of machine learning?
  💡 Querying with first variation...
  [Answer shown]

HOW IT WORKS:
  • Generates 4 alternative phrasings of your question
  • Each variation searches the documents
  • Combines results for better coverage

WHAT TO TRY:
  expand What is a neural network?
  expand how do algorithms work?
  expand explain data science


✅ TEST 7: Multi-Hop Reasoning
──────────────────────────────────────────────────────────────────────────

COMMAND:
  multihop How does machine learning relate to artificial intelligence?

EXPECTED OUTPUT:
  🎯 Decomposing query into steps...
  📝 Original Query: How does machine learning relate to AI?
  🎯 Decomposed into 3 steps:
     1. What is artificial intelligence?
     2. What is machine learning?
     3. How are they connected?
  📚 Retrieving context for each step...
     ✅ Step 1: Retrieved 2 docs
     ✅ Step 2: Retrieved 2 docs
     ✅ Step 3: Retrieved 2 docs
  🔗 Synthesizing final answer...

  💡 SYNTHESIZED ANSWER:
  ════════════════════════════════
  [Comprehensive answer]
  ════════════════════════════════

HOW IT WORKS:
  • Breaks complex query into 3 simpler sub-questions
  • Retrieves data for each step
  • Combines results into one comprehensive answer

WHAT TO TRY:
  multihop How do neural networks relate to machine learning?
  multihop What's the connection between data and AI?


✅ TEST 8: Show Loaded Sources
──────────────────────────────────────────────────────────────────────────

COMMAND:
  sources

EXPECTED OUTPUT:
  📂 LOADED SOURCES (3)
  ════════════════════════════════
  📚 [WIKIPEDIA] Machine Learning
  🌐 [URL] https://en.wikipedia.org/wiki/Machine_learning
  📄 [FILE] test_document.txt


✅ TEST 9: View Conversation History
──────────────────────────────────────────────────────────────────────────

COMMAND:
  history

EXPECTED OUTPUT:
  📜 CONVERSATION HISTORY (4 messages)
  ════════════════════════════════
  [1] 👤 USER
      What is machine learning?...
      Sources: 3 docs

  [2] 🤖 ASSISTANT
      Machine learning is a field of AI...
      Sources: 3 docs

  [3] 👤 USER
      What are neural networks?...

  [4] 🤖 ASSISTANT
      Neural networks are computational models...


✅ TEST 10: Enable Streaming
──────────────────────────────────────────────────────────────────────────

COMMAND:
  streaming

EXPECTED OUTPUT:
  💬 Streaming: ✅ ENABLED

THEN TRY:
  query Explain machine learning

EXPECTED OUTPUT:
  [Answer appears word-by-word in real-time, not all at once]

TOGGLE OFF:
  streaming
  💬 Streaming: ❌ DISABLED


✅ TEST 11: Enable Fact-Checking
──────────────────────────────────────────────────────────────────────────

COMMAND:
  fact-check

EXPECTED OUTPUT:
  🔍 Fact-checking: ✅ ENABLED (checks will run on all answer generation)

THEN TRY:
  query Machine learning was invented in 1956

THEN CHECK RESULTS:
  facts

EXPECTED OUTPUT:
  🔍 FACT-CHECK RESULTS
  📊 Summary:
    Total Facts: 2
    Supported: 2/2 (100%)


✅ TEST 12: View Cache Statistics
──────────────────────────────────────────────────────────────────────────

COMMAND:
  cache

EXPECTED OUTPUT (first time):
  💾 EMBEDDING CACHE STATISTICS
  ════════════════════════════════
  Cache Size:        0/1000 embeddings
  Total Lookups:     0
  Cache Hits:        0
  Cache Misses:      0
  Hit Rate:          0.0%

AFTER MULTIPLE QUERIES:
  💾 EMBEDDING CACHE STATISTICS
  ════════════════════════════════
  Cache Size:        8/1000 embeddings
  Total Lookups:     25
  Cache Hits:        18
  Cache Misses:      7
  Hit Rate:          72.0%


✅ TEST 13: View RAGAS Metrics
──────────────────────────────────────────────────────────────────────────

COMMAND:
  metrics

EXPECTED OUTPUT:
  📊 RAGAS EVALUATION METRICS (3 evaluations)
  ════════════════════════════════
  📈 Summary:
    Total Queries: 3
    Avg Confidence: 82.4%
    Total Docs Retrieved: 9

  📋 Recent Results:
  ┌─────────────────┬────────────┬─────────┐
  │ Query           │ Confidence │ Docs    │
  ├─────────────────┼────────────┼─────────┤
  │ What is ML?     │ 85.2%      │ 3       │
  │ How does NN...  │ 78.9%      │ 3       │
  │ Explain AI...   │ 82.9%      │ 3       │
  └─────────────────┴────────────┴─────────┘


✅ TEST 14: Save Conversation
──────────────────────────────────────────────────────────────────────────

COMMAND:
  save my_conversation

EXPECTED OUTPUT:
  ✅ Saved to my_conversation.json

Then you can find: my_conversation.json in the project folder


✅ TEST 15: Clear History
──────────────────────────────────────────────────────────────────────────

COMMAND:
  clear

EXPECTED OUTPUT:
  ✅ Conversation history cleared

Then verify with:
  history
  (should show empty)


✅ TEST 16: Show Help
──────────────────────────────────────────────────────────────────────────

COMMAND:
  help

EXPECTED OUTPUT:
  [Shows all available commands and usage]


═══════════════════════════════════════════════════════════════════════════════

COMPLETE TEST SEQUENCE (Copy paste in order):
═══════════════════════════════════════════════════════════════════════════════

1. load wikipedia "Machine Learning"
2. query What is machine learning?
3. expand What are types of machine learning?
4. sources
5. history
6. streaming
7. query What is supervised learning?
8. streaming
9. fact-check
10. query Deep learning was invented in 2020
11. facts
12. cache
13. multihop How do neural networks relate to machine learning?
14. metrics
15. save my_test
16. clear
17. history
18. exit


═══════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING:
═══════════════════════════════════════════════════════════════════════════════

❌ "Failed to load from wikipedia"
   FIX: Check internet connection, try different Wikipedia page name

❌ "No module named 'wikipediaapi'"
   FIX: Run: ./venv/bin/pip install wikipediaapi

❌ "Error loading URL"
   FIX: Check if URL is accessible, try a working URL

❌ "File not found"
   FIX: Make sure file exists. Use full path if needed:
        load /path/to/file.txt

❌ "No content to chunk from"
   FIX: The source might be empty, try a different source

❌ "No relevant documents found"
   FIX: Load data first before querying

❌ Command not recognized
   FIX: Check spelling, use lowercase, and include proper format


═══════════════════════════════════════════════════════════════════════════════

LOAD FEATURE DETAILS:
═══════════════════════════════════════════════════════════════════════════════

FORMAT:
  load <source> [collection_name]

SOURCE TYPES (auto-detected):

  1. Wikipedia Page:
     load wikipedia "Page Name"
     load wikipedia "Albert Einstein"
     • Any string without URL format is treated as Wikipedia search

  2. HTTP/HTTPS URL:
     load https://example.com/article
     load https://en.wikipedia.org/wiki/AI
     • URLs are scraped using BeautifulSoup
     • Automatically removes scripts/styles

  3. Local File (.txt, .md):
     load myfile.txt
     load /path/to/file.txt
     load notes.md
     • Reads from local filesystem

  4. PDF File (.pdf):
     load mydocument.pdf
     load /path/to/file.pdf
     • Extracts text from all pages using PyPDF2

OPTIONAL COLLECTION NAME:
  load wikipedia "Python" my_python_collection
  • By default: auto-generated from source name
  • Override by providing custom name
  • Collection stores chunked data in ChromaDB


═══════════════════════════════════════════════════════════════════════════════

WHAT HAPPENS WHEN YOU LOAD:
═══════════════════════════════════════════════════════════════════════════════

Step 1: Detect Source Type
  ↓
Step 2: Fetch Content
  • Wikipedia: Fetch from Wikipedia API
  • URL: Scrape and extract text
  • File: Read local file
  ↓
Step 3: Chunk Content
  • Split into manageable pieces (500-800 tokens each)
  • Adaptive sizing based on content type
  ↓
Step 4: Create Embeddings
  • Convert chunks to embeddings
  • Store in ChromaDB
  ↓
Step 5: Ready to Query
  • You can now query this data
  • System will find relevant chunks
  • Generate answers from retrieved content

═══════════════════════════════════════════════════════════════════════════════

THAT'S IT! You now have a complete guide to test all features.

Start with: python3 launch.py
Then copy-paste the commands above to test everything!

═══════════════════════════════════════════════════════════════════════════════
