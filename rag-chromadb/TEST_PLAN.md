# 🧪 RAG System - Comprehensive Test Plan

## Overview
This document provides step-by-step instructions to test all Phase 1 and Phase 2 features of the Advanced RAG System.

**Test Environment**: macOS, Python 3.13, Virtual Environment active
**Total Estimated Time**: 30-45 minutes

---

## ✅ PRE-TEST CHECKLIST

- [ ] Virtual environment activated: `source venv/bin/activate`
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Environment test passed: `python test_environment.py`
- [ ] `.env` file configured with OpenAI credentials (or using local LLM)
- [ ] ChromaDB directory exists: `./chroma_db/`
- [ ] Internet connection available (for Wikipedia/URL loading)

---

## 🎯 TEST SECTION 1: INITIALIZATION & CORE SETUP

### Test 1.1: Application Start-Up
**Objective**: Verify the application initializes without errors

**Steps**:
```bash
cd /Users/haint/Desktop/Repository/generative-ai-learning/rag-chromadb
source venv/bin/activate
python rag-chromadb.py
```

**Expected Output**:
```
================================================================================
🚀 Advanced RAG System: Phase 2 Enhanced
   Hybrid Search + RAGAS + Query Expansion + Multi-hop + Adversarial Testing
================================================================================

✨ PHASE 1 FEATURES:
  ✓ Hybrid Search (BM25 + Semantic)
  ✓ RAGAS Evaluation Metrics
  ...

✅ Initialized RAG System with conversation ID: 20260225_214826
✅ Hybrid Search Engine + RAGAS Evaluator initialized
✅ Query Expansion + Multi-hop Reasoning + Adversarial Testing initialized
```

**Success Criteria**:
- ✅ Application starts without errors
- ✅ All Phase 1 and Phase 2 features listed
- ✅ Conversation ID generated
- ✅ System ready for input (shows "❓ Enter command or ask a question:")

**Pass/Fail**: _______

---

### Test 1.2: Help Command
**Objective**: Verify command documentation is accessible

**Steps**:
```
❓ Enter command or ask a question: help
```

**Expected Output**: List of all available commands with descriptions

**Success Criteria**:
- ✅ Shows core commands (load, sources, history, metrics, etc.)
- ✅ Shows Phase 2 commands (expand, multihop, test, etc.)
- ✅ Clear descriptions for each command

**Pass/Fail**: _______

---

## 🌐 TEST SECTION 2: PHASE 1 - CORE RAG FEATURES

### Test 2.1: Load Wikipedia Source
**Objective**: Test Wikipedia source loading and adaptive chunking

**Steps**:
```
load Cristiano Ronaldo
```

**Expected Output**:
```
✅ Successfully loaded 3 chunks from Cristiano Ronaldo
   Source Type: WIKIPEDIA
   Collection: cristiano_ronaldo
   Chunking Strategy: Adaptive
```

**Success Criteria**:
- ✅ Source loads successfully
- ✅ Shows number of chunks created
- ✅ Displays source type
- ✅ Creates ChromaDB collection
- ✅ No errors in loading process

**Pass/Fail**: _______

---

### Test 2.2: Load URL Source (Optional)
**Objective**: Test web content loading via URL

**Steps**:
```
load https://en.wikipedia.org/wiki/Albert_Einstein
```

**Expected Output**:
```
✅ Successfully loaded X chunks from https://en.wikipedia.org/wiki/Albert_Einstein
   Source Type: URL
   Collection: en_wikipedia_org_wiki_albert_einstein
```

**Success Criteria**:
- ✅ URL detected correctly
- ✅ Web content scraped successfully
- ✅ Content chunked appropriately
- ✅ Metadata includes URL source

**Pass/Fail**: _______

---

### Test 2.3: Load Local File (Optional)
**Objective**: Test local file loading

**Steps**:
1. Create test file: `echo "Einstein was a theoretical physicist." > test_article.txt`
2. Load it: `load test_article.txt`

**Expected Output**:
```
✅ Successfully loaded 1 chunks from test_article.txt
   Source Type: FILE
   Collection: test_article_txt
```

**Success Criteria**:
- ✅ Local file detected and loaded
- ✅ Content properly chunked
- ✅ Source type marked as FILE

**Pass/Fail**: _______

---

### Test 2.4: Show Loaded Sources
**Objective**: Verify source tracking

**Steps**:
```
sources
```

**Expected Output**:
```
================================================================================
📂 LOADED SOURCES
================================================================================
📚 [WIKIPEDIA] Cristiano Ronaldo
🌐 [URL] https://en.wikipedia.org/wiki/Albert_Einstein
📄 [FILE] test_article.txt
================================================================================
```

**Success Criteria**:
- ✅ All loaded sources displayed
- ✅ Source type indicators shown
- ✅ Multiple sources tracked correctly

**Pass/Fail**: _______

---

### Test 2.5: Basic Query with RAGAS Evaluation
**Objective**: Test standard RAG query with evaluation metrics

**Steps**:
```
What were Cristiano Ronaldo's major achievements in football?
```

**Expected Output**:
```
================================================================================
💡 ANSWER
================================================================================
[Detailed answer about Einstein's contributions]

================================================================================
📚 SOURCES & CONTEXT (3 chunks retrieved)
================================================================================
[Source citations...]

================================================================================
📊 METADATA
================================================================================
  Confidence Score: 85.3%
  Source Types Used: WIKIPEDIA
  Conversation ID: 20260225_214826
  Total Messages in History: 2

================================================================================
🎯 RAGAS EVALUATION METRICS
================================================================================
RAGAS Metrics:
  Context Relevance:  89%
  Answer Relevance:   92%
  Faithfulness:       85%
  ─────────────────────────
  Overall RAG Score:  88%
================================================================================
```

**Success Criteria**:
- ✅ Retrieves relevant information from loaded sources
- ✅ Generates coherent answer
- ✅ Shows RAGAS metrics (3 components)
- ✅ Overall RAG score computed correctly
- ✅ Confidence score displayed
- ✅ Sources cited with type information

**Pass/Fail**: _______

---

### Test 2.6: Conversation History & Context Awareness
**Objective**: Test multi-turn conversation with context

**Steps**:
```
What were his major theories?
```
Then ask:
```
How did those theories change our understanding of space and time?
```

**Expected Output**:
- First answer discusses theories
- Second answer references previous discussion contextually
- Both stored in conversation history

**Then run**:
```
history
```

**Expected Output**:
```
================================================================================
📜 CONVERSATION HISTORY
================================================================================

[1] 👤 USER (2026-02-25T21:48:26...)
    Message: What were Cristiano Ronaldo's major achievements in football?
    Sources: [WIKIPEDIA] Cristiano Ronaldo

[2] 🤖 ASSISTANT (Confidence: 85%)
    Message: Einstein's major contributions include...
    ...

[3] 👤 USER
    Message: What were his major theories?

[4] 🤖 ASSISTANT (Confidence: 88%)
    Message: His major theories include...
```

**Success Criteria**:
- ✅ Both queries answered
- ✅ Second answer shows context awareness
- ✅ Conversation history shows all exchanges
- ✅ Timestamps recorded
- ✅ Confidence scores displayed
- ✅ Sources tracked throughout

**Pass/Fail**: _______

---

### Test 2.7: View Evaluation Metrics Summary
**Objective**: Test RAGAS metrics aggregation

**Steps**:
```
metrics
```

**Expected Output**:
```
================================================================================
📊 RAGAS EVALUATION METRICS
================================================================================

📈 OVERALL METRICS (from 3 evaluations):
  Context Relevance:  87% ⭐
  Answer Relevance:   90% ⭐
  Faithfulness:       86% ⭐
  ─────────────────────────
  Average RAG Score:  87% 🎯

📚 RETRIEVAL METHODS:
  HYBRID: 3 times

🔍 RECENT EVALUATIONS:
╒══════════════════════════════════╤══════════════╤───────────╤═════════╕
│ Query                            │ Method       │ RAG Score │ Chunks  │
╞══════════════════════════════════╪══════════════╪═══════════╪═════════╡
│ What were his major theories?    │ hybrid       │ 88%       │ 3       │
├──────────────────────────────────┼──────────────┼───────────┼─────────┤
│ What were Cristiano Ronaldo's│ hybrid       │ 87%       │ 3       │
```

**Success Criteria**:
- ✅ Averages calculated correctly
- ✅ Shows metrics from all evaluations
- ✅ Retrieval methods tracked
- ✅ Recent evaluations displayed in table format
- ✅ No errors in aggregation

**Pass/Fail**: _______

---

## 🚀 TEST SECTION 3: PHASE 2 - ADVANCED FEATURES

### Test 3.1: Query Expansion
**Objective**: Test query variation generation for improved retrieval

**Steps**:
```
expand What was Einstein's early life like?
```

**Expected Output**:
```
🔄 Generating 4 query variations...
✅ Generated 4 variations

📋 Query Expansion Results:
  1. What was Ronaldo's early life like?
  2. Ronaldo's childhood and upbringing - biographical details
  3. How did Ronaldo's formative years shape his thinking?
  4. Personal history and background of Cristiano Ronaldo during youth
  5. What were the key events in Ronaldo's youth and family life?
```

**Success Criteria**:
- ✅ Generates 4 query variations
- ✅ Variations are diverse (paraphrases, synonyms, perspectives)
- ✅ Each variation focuses on same core question
- ✅ Original query included as first variation
- ✅ No errors in generation

**Pass/Fail**: _______

---

### Test 3.2: Query Expansion Retrieval
**Objective**: Verify query expansion improves retrieval coverage

**Steps**: Query expansion automatically retrieves with all variations

**Expected Output from step 3.1**:
- Should show combined retrieval from all 4 variations
- Retrieved documents deduplicated and ranked

**Then check**:
```
expansions
```

**Expected Output**:
```
================================================================================
🔄 QUERY EXPANSION HISTORY
================================================================================

[1] Original: What was Ronaldo's early life like?
    Generated 4 variations:
      1. What was Ronaldo's early life like?
      2. Ronaldo's childhood and upbringing - biographical details
      3. How did Ronaldo's formative years shape his thinking?
      4. Personal history and background of Cristiano Ronaldo during youth
```

**Success Criteria**:
- ✅ expands query into 4 variations
- ✅ Retrieves from all variations
- ✅ Combines and deduplicates results
- ✅ Stores expansion history
- ✅ History can be viewed

**Pass/Fail**: _______

---

### Test 3.3: Multi-hop Reasoning
**Objective**: Test complex query decomposition into steps

**Steps**:
```
multihop How did Einstein's theories revolutionize our understanding of physics?
```

**Expected Output**:
```
🎯 Processing query with multi-hop reasoning (3 steps)...

Step 1/3: What were Einstein's major theories?
[retrieves and synthesizes answer]

Step 2/3: What was physics like before Einstein?
[retrieves and synthesizes answer]

Step 3/3: How did Einstein's ideas change scientific thinking?
[retrieves and synthesizes answer]

🔗 Synthesizing multi-hop answer...

✅ Multi-hop reasoning complete (3 steps, confidence: 87%)

================================================================================
💡 ANSWER
================================================================================
[Synthesized comprehensive answer combining all 3 steps]
```

**Success Criteria**:
- ✅ Query decomposed into 3 logical sub-questions
- ✅ Each step retrieves relevant information
- ✅ Final answer synthesizes all steps coherently
- ✅ Confidence score calculated from all steps
- ✅ RAGAS metrics computed for final answer

**Pass/Fail**: _______

---

### Test 3.4: View Multi-hop Results
**Objective**: Verify multi-hop reasoning results are stored

**Steps**:
```
multihop-results
```

**Expected Output**:
```
================================================================================
🎯 MULTI-HOP REASONING RESULTS
================================================================================

Query: How did Einstein's theories revolutionize our understanding of physics?
Confidence: 87%
Steps: 3

  Step 1: What were Einstein's major theories?
    Reasoning: Einstein proposed the theory of relativity...
    Retrieved: [document snippets]
    Confidence: 86%

  Step 2: What was physics like before Einstein?
    Reasoning: Classical Newtonian physics dominated...
    Retrieved: [document snippets]
    Confidence: 88%

  Step 3: How did Einstein's ideas change scientific thinking?
    Reasoning: Einstein's work fundamentally changed...
    Retrieved: [document snippets]
    Confidence: 87%
```

**Success Criteria**:
- ✅ Shows decomposed steps
- ✅ Displays reasoning for each step
- ✅ Retrieved documents shown per step
- ✅ Confidence score calculated per step
- ✅ Overall confidence computed

**Pass/Fail**: _______

---

### Test 3.5: Adversarial Testing Suite
**Objective**: Test RAG robustness with edge cases

**Steps**:
```
test
```

**Expected Output**:
```
🧪 Running adversarial test suite...

Running test ambig_001: ambiguous
✅ PASS - System handled ambiguous query gracefully

Running test ambig_002: ambiguous
✅ PASS - System recognized missing context

Running test noans_001: no_answer
✅ PASS - System acknowledged question has no valid answer

Running test noans_002: no_answer
⚠️  PARTIAL - System provided best effort answer

Running test conflict_001: conflicting
✅ PASS - System identified conflicting statements

Running test edge_001: edge_case
✅ PASS - Empty query handled gracefully

Running test edge_002: edge_case
✅ PASS - Long query handled without crashing

Running test edge_003: edge_case
✅ PASS - Special characters processed correctly

================================================================================
🧪 ADVERSARIAL TEST RESULTS
================================================================================

📊 SUMMARY:
  Total Tests: 8
  Passed: 7 (87.5%)
  Failed: 1 (12.5%)

📋 DETAILED RESULTS:
╒═══════════╤════════════════╤──────────────────────╤════════════╕
│ Test ID   │ Type           │ Query                │ Status     │
╞═══════════╪════════════════╪══════════════════════╪════════════╡
│ ambig_001 │ ambiguous      │ What about design?   │ ✅ PASS    │
├───────────┼────────────────┼──────────────────────┼────────────┤
│ ambig_002 │ ambiguous      │ Is it better?        │ ✅ PASS    │
├───────────┼────────────────┼──────────────────────┼────────────┤
│ noans_001 │ no_answer      │ What color is 7?     │ ✅ PASS    │
├───────────┼────────────────┼──────────────────────┼────────────┤
│ noans_002 │ no_answer      │ Tell me about -5000  │ ⚠️  PARTIAL│
...

⚠️ FAILURES (1):
  Test: noans_002 - Expected system to return "no information" but got an answer
```

**Success Criteria**:
- ✅ All 8 tests run without crashing
- ✅ Pass/fail status determined for each test
- ✅ Summary statistics calculated
- ✅ Detailed results displayed in table
- ✅ Failed tests explained
- ✅ Results saved to file

**Pass/Fail**: _______

---

### Test 3.6: View Adversarial Test Results
**Objective**: Verify test results persistence

**Steps**:
```
test-results
```

**Expected Output**: Same as Test 3.5, pulled from saved results

**Success Criteria**:
- ✅ Results match previous test run
- ✅ Can be called multiple times without re-running
- ✅ Results file persists

**Pass/Fail**: _______

---

## 🔄 TEST SECTION 4: INTEGRATION & END-TO-END

### Test 4.1: Complete Workflow (5 minutes)
**Objective**: Test all features in realistic workflow

**Steps**:
1. Load a source: `load Marie Curie`
2. Ask question: `Who was Marie Curie and what did she discover?`
3. Expand query: `expand What were Marie Curie's most important scientific achievements?`
4. Multi-hop: `multihop How did Marie Curie's work change radiation science?`
5. View metrics: `metrics`
6. View history: `history`
7. Run tests: `test`
8. Check results: `test-results`

**Success Criteria**:
- ✅ All 8 operations complete without errors
- ✅ Each operation produces expected output
- ✅ Data persists across commands
- ✅ Conversation history accumulates
- ✅ Metrics improve with multiple queries
- ✅ No memory leaks or performance degradation

**Pass/Fail**: _______

---

### Test 4.2: Data Persistence
**Objective**: Verify data saved between sessions

**Steps**:
1. After completing Test 4.1, exit: `quit`
2. Restart application and reload: `python rag-chromadb.py`
3. Check history: `history`
4. Check metrics: `metrics`
5. View expansions: `expansions`

**Expected Output**:
- Previous conversation history loaded
- Metrics from previous queries shown
- Query expansions preserved

**Success Criteria**:
- ✅ Conversation history loads from file
- ✅ Evaluation metrics restored
- ✅ Query expansions not lost
- ✅ Multi-hop results persist
- ✅ Test results remain available

**Pass/Fail**: _______

---

## 📊 TEST SECTION 5: PERFORMANCE & STABILITY

### Test 5.1: Query Performance
**Objective**: Check query processing time

**Steps**:
```
What are the key principles of special relativity?
```

Check the timestamp in the output and note response time.

**Success Criteria**:
- ✅ Query processes in < 10 seconds (with LLM calls)
- ✅ No timeout errors
- ✅ Response quality maintained

**Pass/Fail**: _______

---

### Test 5.2: Memory Stability
**Objective**: Check memory usage doesn't grow excessively

**Steps**:
1. Run 10 queries in succession
2. Observe the application doesn't slow down
3. Final response quality same as first

**Success Criteria**:
- ✅ Response times remain consistent
- ✅ No visible memory growth
- ✅ Application remains responsive
- ✅ No crashes or hangs

**Pass/Fail**: _______

---

### Test 5.3: Error Handling
**Objective**: Test graceful error handling

**Steps**:
1. Try to query without loading a source:
   ```
   What is the meaning of life?
   ```
2. Try invalid commands:
   ```
   invalid_command
   ```
3. Try special characters:
   ```
   @#$%^&*()!?
   ```

**Success Criteria**:
- ✅ Appropriate error messages shown
- ✅ No crashes or stack traces
- ✅ Application continues running
- ✅ Can recover and continue

**Pass/Fail**: _______

---

## 📝 TEST SECTION 6: DOCUMENTATION & HELP

### Test 6.1: Command Help
**Objective**: Verify all commands documented

**Steps**: Review all command descriptions during initial startup

**Success Criteria**:
- ✅ All 14+ commands listed
- ✅ Clear descriptions for each
- ✅ Examples provided for main commands
- ✅ Organization by feature (Phase 1, Phase 2)

**Pass/Fail**: _______

---

## 📋 TEST SUMMARY CHECKLIST

**Core System** (Section 1)
- [ ] Test 1.1: Application Start-Up
- [ ] Test 1.2: Help Command

**Phase 1 Features** (Section 2)
- [ ] Test 2.1: Load Wikipedia Source
- [ ] Test 2.2: Load URL Source (Optional)
- [ ] Test 2.3: Load Local File (Optional)
- [ ] Test 2.4: Show Loaded Sources
- [ ] Test 2.5: Basic Query with RAGAS
- [ ] Test 2.6: Conversation History
- [ ] Test 2.7: View Evaluation Metrics

**Phase 2 Features** (Section 3)
- [ ] Test 3.1: Query Expansion
- [ ] Test 3.2: Query Expansion Retrieval
- [ ] Test 3.3: Multi-hop Reasoning
- [ ] Test 3.4: View Multi-hop Results
- [ ] Test 3.5: Adversarial Testing
- [ ] Test 3.6: View Adversarial Results

**Integration** (Section 4)
- [ ] Test 4.1: Complete Workflow
- [ ] Test 4.2: Data Persistence

**Performance** (Section 5)
- [ ] Test 5.1: Query Performance
- [ ] Test 5.2: Memory Stability
- [ ] Test 5.3: Error Handling

**Documentation** (Section 6)
- [ ] Test 6.1: Command Help

---

## 🎯 FINAL SIGN-OFF

**Tester Name**: _________________________

**Test Date**: _________________________

**Overall Status**: ☐ PASS ☐ PARTIAL PASS ☐ FAIL

**Issues Found**:
```
1.
2.
3.
```

**Notes**:
```


```

**Recommendation**: ☐ Ready for Production ☐ Needs Fixes ☐ Major Issues

---

**Document Version**: 1.0
**Last Updated**: February 25, 2026
**Testing Framework**: Manual comprehensive testing
