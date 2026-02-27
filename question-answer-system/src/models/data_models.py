"""Core data models for RAG system"""
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class RetrievedDocument:
    """Represents a retrieved document chunk with metadata."""
    content: str
    source: str
    source_type: str  # 'wikipedia', 'url', 'file', 'pdf'
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
    conversation_context: Optional[str] = None
    execution_time_ms: Optional[float] = None
    fact_check_results: Optional[List['FactCheckResult']] = None

    def to_dict(self):
        return {
            "answer": self.answer,
            "sources": [asdict(s) for s in self.sources],
            "confidence_score": self.confidence_score,
            "source_types": self.source_types,
            "conversation_context": self.conversation_context,
            "execution_time_ms": self.execution_time_ms,
            "fact_check_results": [r.to_dict() for r in self.fact_check_results] if self.fact_check_results else None,
        }


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
            "query": self.query,
            "answer": self.answer,
            "metrics": self.metrics.to_dict(),
            "retrieval_method": self.retrieval_method,
            "num_chunks_retrieved": self.num_chunks_retrieved,
            "timestamp": self.timestamp,
        }


@dataclass
class FactCheckResult:
    """Result of fact-checking a claim."""
    fact: str
    is_supported: bool
    confidence: float
    evidence: Optional[str] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class QueryExpansion:
    """Query expansion result with variations."""
    original_query: str
    variations: List[str]  # Generated query variations
    expansion_method: str  # 'paraphrase', 'synonym', 'decompose'
    timestamp: str

    def to_dict(self):
        return asdict(self)


@dataclass
class MultiHopStep:
    """Single step in multi-hop reasoning."""
    step_number: int
    subquery: str
    retrieved_docs: List[str]  # Content snippets
    reasoning: str  # Why this step
    relevance_score: float

    def to_dict(self):
        return asdict(self)


@dataclass
class MultiHopResult:
    """Complete multi-hop reasoning result."""
    original_query: str
    steps: List[MultiHopStep]
    final_answer: str
    total_confidence: float
    timestamp: str

    def to_dict(self):
        return {
            'original_query': self.original_query,
            'steps': [step.to_dict() for step in self.steps],
            'final_answer': self.final_answer,
            'total_confidence': self.total_confidence,
            'timestamp': self.timestamp
        }


@dataclass
class AdversarialTestCase:
    """Adversarial test case for RAG system."""
    test_id: str
    query: str
    test_type: str  # 'ambiguous', 'no_answer', 'conflicting', 'edge_case'
    expected_behavior: str  # What should happen
    result: Optional[str] = None
    passed: Optional[bool] = None
    error_message: Optional[str] = None
    timestamp: str = ""

    def to_dict(self):
        return asdict(self)
