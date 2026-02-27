"""Fact checking module for verifying claims in answers"""
import re
from typing import List, Tuple
from src.models import FactCheckResult
from src.config import get_config
from src.utils import get_logger
from openai import OpenAI
from datetime import datetime

logger = get_logger()


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
