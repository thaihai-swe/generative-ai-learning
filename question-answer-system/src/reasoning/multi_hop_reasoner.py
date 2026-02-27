"""Multi-hop reasoning module for decomposing and synthesizing complex queries"""
from typing import List, Dict
from src.config import get_config
from src.utils import get_logger
from openai import OpenAI

logger = get_logger()


class MultiHopReasoner:
    """Performs multi-hop reasoning by breaking complex queries into steps"""

    @staticmethod
    def decompose(query: str, max_steps: int = 3) -> List[str]:
        """Decompose complex query into sub-questions"""
        logger.info(f"🎯 Decomposing query into {max_steps} steps...")

        config = get_config()
        client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )

        try:
            prompt = f"""Break down this complex query into {max_steps} simpler sub-questions that together help answer it.
Each sub-question should build on previous understanding.

Complex Query: {query}

Return ONLY the sub-questions, one per line, without numbering or extra formatting.
Each should be a complete, standalone question."""

            response = client.chat.completions.create(
                model=config.llm.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at decomposing complex questions into simpler steps."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=400
            )

            steps_text = response.choices[0].message.content.strip()
            steps = [s.strip() for s in steps_text.split('\n') if s.strip()]
            logger.info(f"✅ Decomposed into {len(steps)} steps")
            return steps[:max_steps]

        except Exception as e:
            logger.warning(f"⚠️ Query decomposition failed: {str(e)}")
            return [query]

    @staticmethod
    def synthesize(query: str, step_results: List[Dict]) -> str:
        """Synthesize final answer from multi-hop step results"""
        logger.info("🔗 Synthesizing multi-hop answer...")

        config = get_config()
        client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )

        try:
            step_summary = "\n".join([
                f"Step {i+1} ({sr.get('subquery', 'N/A')}): {sr.get('answer', '')[:200]}"
                for i, sr in enumerate(step_results)
            ])

            prompt = f"""Based on the following step-by-step reasoning, provide a comprehensive answer to the original query.
Synthesize all the information into a coherent, unified response.

Original Query: {query}

Step-by-step Results:
{step_summary}

Provide a synthesized answer that incorporates all findings."""

            response = client.chat.completions.create(
                model=config.llm.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are synthesizing multi-hop reasoning into a comprehensive answer."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            answer = response.choices[0].message.content.strip()
            logger.info("✅ Synthesized answer from multi-hop reasoning")
            return answer

        except Exception as e:
            logger.warning(f"⚠️ Answer synthesis failed: {str(e)}")
            return ""
