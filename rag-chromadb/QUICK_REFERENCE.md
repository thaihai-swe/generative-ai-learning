# 🚀 RAG System - Quick Reference Guide

## Quick Start

```bash
cd /Users/haint/Desktop/Repository/generative-ai-learning/rag-chromadb
source venv/bin/activate
python rag-chromadb.py
```

---

## 📋 All Commands At A Glance

### Core Commands
| Command         | Purpose                       | Example                  |
| --------------- | ----------------------------- | ------------------------ |
| `load <source>` | Load Wikipedia, URL, or file  | `load Cristiano Ronaldo` |
| `sources`       | Show all loaded sources       | `sources`                |
| `<question>`    | Ask standard RAG query        | `What is relativity?`    |
| `history`       | Show conversation history     | `history`                |
| `metrics`       | Show RAGAS evaluation metrics | `metrics`                |
| `clear`         | Clear conversation history    | `clear`                  |
| `quit`          | Exit application              | `quit`                   |

### Phase 2 Advanced Commands
| Command            | Purpose                        | Example                                       |
| ------------------ | ------------------------------ | --------------------------------------------- |
| `expand <query>`   | Query expansion (4 variations) | `expand What was Einstein's early life?`      |
| `multihop <query>` | Multi-hop reasoning (3 steps)  | `multihop How did relativity change physics?` |
| `expansions`       | Show query expansion history   | `expansions`                                  |
| `multihop-results` | Show multi-hop results         | `multihop-results`                            |
| `test`             | Run adversarial test suite     | `test`                                        |
| `test-results`     | Show test results              | `test-results`                                |

---

## 🎯 Quick Test Matrix

### 5-Minute Test
1. `load Cristiano Ronaldo`
2. `Ask about his football career`
3. `metrics`
4. `quit`

### 10-Minute Test
1. `load Cristiano Ronaldo`
2. `Ask about his achievements`
3. `expand What was his early life like?`
4. `multihop How did he become famous?`
5. `metrics`
6. `quit`

### 20-Minute Full Test
1. `load Cristiano Ronaldo`
2. `What were his career achievements?`
3. `expand What was his early life like?`
4. `multihop How did he become a legend?`
5. `test`
6. `test-results`
7. `metrics`
8. `history`
9. `quit`

---

## 📊 Key Metrics to Check

After running queries, run `metrics` to see:

```
Average Scores (from all evaluations):
- Context Relevance: __ %  (retrieval quality)
- Answer Relevance: __ %   (answer quality)
- Faithfulness: __ %       (no hallucinations)
- RAG Score: __ %          (overall)

Benchmark:
✅ Excellent: >90%
✅ Good: 80-90%
⚠️  Okay: 70-80%
❌ Needs Work: <70%
```

---

## 💡 Demo Scripts (Copy & Paste)

### The 2-Min Demo (Impact)
```
load Cristiano Ronaldo
What are his major achievements?
metrics
```

**Key Point**: "Notice the RAGAS metrics showing context relevance, answer relevance, and faithfulness—these measure whether the system is actually retrieving good information and not hallucinating."

---

### The 5-Min Demo (Depth)
```
load Cristiano Ronaldo
What are his career highlights?
expand How did he revolutionize football?
multihop What was the relationship between his skills and his success?
metrics
```

**Key Points**:
1. Standard RAG query with evaluation
2. Query expansion shows retrieval coverage
3. Multi-hop reasoning shows complex problem-solving
4. Metrics show end-to-end quality

---

### The 10-Min Demo (Production-Ready)
```
load Cristiano Ronaldo
load Lionel Messi
What are his career highlights?
expand How did he revolutionize football?
multihop What was the relationship between Ronaldo and Messi's contributions?
test
test-results
metrics
history
```

**Key Points**:
1. Multi-source loading
2. Query types (standard, expansion, multi-hop)
3. Adversarial testing (robustness)
4. Quality metrics (quantified)
5. Conversation history (auditability)
6. Complete end-to-end workflow

---

## 🔧 Troubleshooting

| Issue                  | Solution                                                |
| ---------------------- | ------------------------------------------------------- |
| "No sources loaded"    | Use `load <source>` first                               |
| "No metrics available" | Need to run at least one query first                    |
| "API key error"        | Set `.env` with `OPEN_AI_API_KEY`                       |
| "Slow responses"       | Normal if using LLM evaluation (evaluator + generation) |
| "Empty history"        | Run at least one query first                            |

---

## 📁 File Structure (Auto-Generated)

These files are created automatically as you use the system:

```
conversation_history.json       → All Q&A exchanges
evaluation_metrics.json         → RAGAS scores for each query
query_expansions.json          → Expanded queries and variations
multihop_results.json          → Multi-step reasoning results
adversarial_test_results.json  → Test pass/fail results
```

---

## 🎬 Scenario-Based Commands

### For Showing Retrieval Quality
```
load [source]
[Ask natural question]
metrics
expand [same question]
metrics
```
**Show**: Query expansion improves scores

### For Showing Reasoning Ability
```
multihop [complex question with multiple parts]
multihop-results
```
**Show**: How complex questions are decomposed

### For Showing Robustness
```
test
test-results
```
**Show**: Edge case handling and pass rate

### For Showing Data Integration
```
load [source A]
load [source B]
[Ask question requiring both sources]
```
**Show**: Multi-source synthesis

---

## 💻 Interview Question Triggers

When someone asks...

**"How does retrieval work?"** →
```
expand What was Einstein's biggest achievement?
```
Then show the 4 variations generated

**"Can it handle complex questions?"** →
```
multihop How did Einstein's work lead to nuclear physics?
```
Then show the 3-step decomposition

**"How do you ensure quality?"** →
```
metrics
```
Then explain RAGAS metrics

**"What about hallucinations?"** →
```
multihop [obviously false premise question]
metrics
```
Then show how faithfulness catches it

**"How is this different?"** →
```
history
```
Then show source attribution and conversation context

---

## 📊 Expected Outputs Cheat Sheet

### When You See This | It Means This
| Output                 | Interpretation                     |
| ---------------------- | ---------------------------------- |
| Context Relevance: 85% | Good retrieval quality             |
| Answer Relevance: 90%  | LLM stays on-topic                 |
| Faithfulness: 82%      | Some hallucination detected        |
| Passed: 7/8 tests      | Robust system, one edge case issue |
| Confidence: 75%        | Answer is less certain             |
| ✓ 3 chunks retrieved   | Got good information diversity     |

---

## 🚨 Warning Signs

| Warning                 | Action                                      |
| ----------------------- | ------------------------------------------- |
| Faithfulness <75%       | System hallucinates, needs better prompting |
| Confidence <60%         | Answer is unreliable, check retrieval       |
| Tests <70% passing      | Edge cases not handled well                 |
| Same metrics repeatedly | Query quality may not matter, check setup   |

---

## 🎯 Talking Point Templates

### On Hybrid Search
> "We use 70% semantic search (understanding meaning) and 30% keyword search (exact matches). Neither alone is sufficient, but together they cover 95% of use cases."

### On RAGAS Metrics
> "We measure three things: Is the context relevant? Is the answer on-topic? Is the answer grounded in the context? These three metrics catch most RAG failure modes."

### On Query Expansion
> "Instead of searching once, we generate four phrasings and search with all of them. Different ways of asking get different results. Combining them improves coverage."

### On Multi-hop Reasoning
> "Complex questions need multiple steps. We break them down, answer each part independently with retrieval, then synthesize the final answer. This is more robust than one-shot generation."

### On Adversarial Testing
> "We test systematically: ambiguous queries, no-answer questions, edge cases. 87% pass rate means we're solid but not perfect. The failures tell us where to improve."

---

## 📈 Success Metrics for Portfolio

Track these as you run demos:

```
Overall RAG Score:     ___ %  (Target: >85%)
Average Confidence:    ___ %  (Target: >80%)
Adversarial Pass Rate: ___ %  (Target: >85%)
Sources Used Per Query: ___   (Target: 3+)
Commands Executed:     ___    (Target: 6+ in demo)
User Satisfaction:     ___ %  (Target: >90%)
```

---

## 🎬 Recording a Demo (For Portfolio Video)

1. Start with clean terminal: `clear`
2. Show: `cat SETUP_COMPLETE.md` (quick intro)
3. Run: `python rag-chromadb.py` (show startup)
4. Execute: 10-minute demo sequence
5. End with: `quit` (graceful exit)
6. Total time: ~5 minutes of content

**Narration Points**:
- Problem: Standard chatbots don't cite sources
- Solution: This RAG system grounds answers in documents
- Differentiation: Hybrid search + multi-hop reasoning + quality metrics
- Impact: Production-ready, not academic

---

## ⏱️ Time Planning

- **Quick validation**: 3 minutes
- **Technical interview**: 10 minutes
- **Full demo**: 15 minutes
- **Full test suite**: 30 minutes
- **Deep dive**: 45+ minutes

Choose based on time available and audience technical level.

---

## 🎓 Learning Resources

To understand the system better:
- **SETUP_COMPLETE.md**: Environment and features overview
- **TEST_PLAN.md**: Detailed test procedures
- **DEMO_SCENARIOS.md**: Complete demo narratives
- **rag-chromadb.py**: Source code comments throughout

---

## 🎯 Final Checklist Before Demo

- [ ] Venv activated
- [ ] Sources pre-loaded (if planned)
- [ ] Network connection working
- [ ] Terminal at readable zoom level
- [ ] `.env` file configured
- [ ] Know your talking points
- [ ] Have backup demo offline option
- [ ] Test metrics showing (at least 1 query run)

---

## 💡 Pro Tips

1. **Pre-load sources** before demo starts (saves 30 seconds)
2. **Have 2-3 questions** pre-planned (shows confidence)
3. **Explain metrics** even if people don't ask (shows deep knowledge)
4. **Show the code** briefly to prove it's real
5. **Run tests** if time permits (proves robustness)
6. **Save metrics output** for later reference

---

**Quick Version**: This document.
**Full Version**: See DEMO_SCENARIOS.md
**Testing**: See TEST_PLAN.md
**Setup**: See SETUP_COMPLETE.md

---

*Last updated: February 25, 2026*
*Version: 1.0 - Production Ready*
