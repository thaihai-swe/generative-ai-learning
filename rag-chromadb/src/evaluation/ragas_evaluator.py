"""RAGAS evaluation implementation"""
from src.evaluation.base import Evaluator
from src.models import RAGASMetrics
from src.config import get_config
from src.utils import get_logger
from openai import OpenAI

logger = get_logger()


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
