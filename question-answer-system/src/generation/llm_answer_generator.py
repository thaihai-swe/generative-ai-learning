"""LLM-based answer generator implementation"""
from typing import List, Iterator
from src.models import RetrievedDocument
from src.config import get_config
from src.utils import get_logger
from openai import OpenAI
from .answer_generator import AnswerGenerator

logger = get_logger()


class LLMAnswerGenerator(AnswerGenerator):
    """Generate answers using LLM"""

    def __init__(self):
        config = get_config()
        self.client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )
        self.model = config.llm.model_name
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens

    def generate(self, query: str, context: List[RetrievedDocument]) -> str:
        """Generate answer from context (buffered)"""
        system_prompt = """You are a knowledgeable and helpful assistant. Based on the provided context,
answer the user's question accurately and concisely. If the context doesn't contain enough information,
acknowledge this and provide your best response based on what is available."""

        context_text = self._build_context(context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"{context_text}\n\nQuestion: {query}"
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Unable to generate answer."

    def generate_streaming(self, query: str, context: List[RetrievedDocument]) -> Iterator[str]:
        """Stream answer token-by-token"""
        system_prompt = """You are a knowledgeable and helpful assistant. Based on the provided context,
answer the user's question accurately and concisely."""

        context_text = self._build_context(context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"{context_text}\n\nQuestion: {query}"
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"Error: {str(e)}"

    @staticmethod
    def _build_context(docs: List[RetrievedDocument]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Source {i} ({doc.source}):\n{doc.content}\n")
        return "\n".join(context_parts)
