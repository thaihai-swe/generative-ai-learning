"""Evaluation layer - quality assessment"""
from abc import ABC, abstractmethod
import re
from typing import List, Tuple
from src.models import RAGASMetrics, FactCheckResult
from src.config import get_config
from src.utils import get_logger
from openai import OpenAI
from datetime import datetime

logger = get_logger()


class Evaluator(ABC):
    """Abstract evaluator interface"""

    @abstractmethod
    def evaluate(self, query: str, context: str, answer: str) -> RAGASMetrics:
        """Evaluate RAG output quality"""
        pass


class RAGASEvaluator(Evaluator):
    """Evaluates RAG quality using RAGAS-inspired metrics"""

    def __init__(self):
        config = get_config()
        self.client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )
        self.model = config.llm.model_name

    def evaluate(self, query: str, context: str, answer: str) -> RAGASMetrics:
        """Perform full RAGAS evaluation"""
        logger.info("📊 Running RAGAS evaluation...")

        context_relevance = self._evaluate_context_relevance(query, context)
        answer_relevance = self._evaluate_answer_relevance(query, answer)
        faithfulness = self._evaluate_faithfulness(context, answer)
        rag_score = self._compute_rag_score(context_relevance, answer_relevance, faithfulness)

        return RAGASMetrics(
            context_relevance=context_relevance,
            answer_relevance=answer_relevance,
            faithfulness=faithfulness,
            rag_score=rag_score
        )

    def _evaluate_context_relevance(self, query: str, context: str) -> float:
        """Context Relevance: Is retrieved context relevant to query?"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating the relevance of document context to a query. Respond with only a number from 0 to 10."
                    },
                    {
                        "role": "user",
                        "content": f"""Query: {query}\n\nContext: {context[:500]}\n\nOn a scale of 0-10, how relevant is this context to the query?\nOnly provide a number."""
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

    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Answer Relevance: Does the answer address the query?"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating if an answer directly addresses a query. Respond with only a number from 0 to 10."
                    },
                    {
                        "role": "user",
                        "content": f"""Query: {query}\n\nAnswer: {answer[:500]}\n\nOn a scale of 0-10, how well does this answer address the query?\nOnly provide a number."""
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

    def _evaluate_faithfulness(self, context: str, answer: str) -> float:
        """Faithfulness: Is the answer grounded in the provided context?"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating if an answer is grounded in provided context. Respond with only a number from 0 to 10."
                    },
                    {
                        "role": "user",
                        "content": f"""Context: {context[:500]}\n\nAnswer: {answer[:500]}\n\nOn a scale of 0-10, how much of the answer is supported by the context?\nOnly provide a number."""
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
    def _compute_rag_score(context_relevance: float, answer_relevance: float, faithfulness: float) -> float:
        """Compute overall RAG score as weighted average"""
        weights = [0.30, 0.35, 0.35]
        scores = [context_relevance, answer_relevance, faithfulness]
        return sum(s * w for s, w in zip(scores, weights))


class FactChecker:
    """Fact-checking module to verify claims in generated answers"""

    def __init__(self):
        config = get_config()
        self.client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )
        self.model = config.llm.model_name

    @staticmethod
    def extract_facts(text: str) -> List[str]:
        """Extract fact claims from text"""
        sentences = re.split(r'[.!?]+', text)
        facts = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return facts

    def _check_fact_against_context(self, fact: str, context: str) -> Tuple[bool, str, float]:
        """Check if a fact is supported by the context"""
        try:
            prompt = f"""Based on the provided context, determine if the following statement is supported, contradicted, or unknown:

Statement: "{fact}"

Context:
{context}

Respond in this exact format:
SUPPORTED|CONTRADICTED|UNKNOWN
Confidence: [0-100]
Evidence: [brief explanation]"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            result_text = response.choices[0].message.content
            lines = result_text.split('\n')

            verdict = "UNKNOWN"
            confidence = 50
            evidence = ""

            if lines:
                verdict = lines[0].split('|')[0].strip() if '|' in lines[0] else lines[0].strip()
            if len(lines) > 1:
                conf_line = lines[1].split(':')[-1].strip()
                confidence = int(''.join(filter(str.isdigit, conf_line)) or '50') / 100.0
            if len(lines) > 2:
                evidence = lines[2].split(':')[-1].strip() if ':' in lines[2] else ''

            is_supported = verdict == "SUPPORTED"
            return is_supported, evidence, confidence

        except Exception as e:
            logger.warning(f"⚠️ Fact-checking failed: {str(e)}")
            return False, str(e), 0.0

    def check_answer(self, answer: str, context: str) -> List[FactCheckResult]:
        """Check all facts in an answer against context"""
        logger.info("🔍 Running fact-check...")
        facts = self.extract_facts(answer)
        results = []

        for fact in facts[:5]:  # Check max 5 facts to save tokens
            is_supported, evidence, confidence = self._check_fact_against_context(fact, context)
            result = FactCheckResult(
                fact=fact,
                is_supported=is_supported,
                supporting_evidence=evidence,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            results.append(result)

        logger.info(f"✅ Fact-check complete ({len(results)} facts checked)")
        return results


__all__ = ["Evaluator", "RAGASEvaluator", "FactChecker"]
