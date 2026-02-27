"""Adversarial testing module for identifying RAG system weaknesses"""
from typing import List, TYPE_CHECKING
from src.models import AdversarialTestCase
from src.config import get_config
from src.utils import get_logger
from datetime import datetime

if TYPE_CHECKING:
    from src.core import RAGSystem

logger = get_logger()


class AdversarialTestSuite:
    """Generates and runs adversarial tests to identify RAG system weaknesses"""

    @staticmethod
    def generate_test_cases() -> List[AdversarialTestCase]:
        """Generate diverse adversarial test cases"""
        logger.info("🧪 Generating adversarial test cases...")

        test_cases = [
            # Ambiguous queries
            AdversarialTestCase(
                test_id="ambig_001",
                query="What about design?",
                test_type="ambiguous",
                expected_behavior="System should ask for clarification or provide multiple interpretations"
            ),
            AdversarialTestCase(
                test_id="ambig_002",
                query="Is it better?",
                test_type="ambiguous",
                expected_behavior="System should recognize missing context and ask for specifics"
            ),

            # No answer exists
            AdversarialTestCase(
                test_id="noans_001",
                query="What color is number 7?",
                test_type="no_answer",
                expected_behavior="System should acknowledge that the question doesn't have a valid answer"
            ),
            AdversarialTestCase(
                test_id="noans_002",
                query="Tell me about events that happened in the year -5000",
                test_type="no_answer",
                expected_behavior="System should state it doesn't have information about this topic"
            ),

            # Conflicting information
            AdversarialTestCase(
                test_id="conflict_001",
                query="Synthesize the following contradictory statements...",
                test_type="conflicting",
                expected_behavior="System should identify and highlight conflicting viewpoints"
            ),

            # Edge cases
            AdversarialTestCase(
                test_id="edge_001",
                query="",  # Empty query
                test_type="edge_case",
                expected_behavior="System should handle gracefully without crashing"
            ),
            AdversarialTestCase(
                test_id="edge_002",
                query="a" * 1000,  # Very long query
                test_type="edge_case",
                expected_behavior="System should handle long inputs gracefully"
            ),
            AdversarialTestCase(
                test_id="edge_003",
                query="!@#$%^&*()",  # Special characters only
                test_type="edge_case",
                expected_behavior="System should handle special characters without crashing"
            ),
        ]

        logger.info(f"✅ Generated {len(test_cases)} test cases")
        return test_cases

    @staticmethod
    def run_test_case(rag_system: "RAGSystem", test_case: AdversarialTestCase) -> AdversarialTestCase:
        """Run a single adversarial test"""
        logger.info(f"Running test {test_case.test_id}: {test_case.test_type}")

        try:
            if not test_case.query.strip():
                test_case.result = "ERROR: Empty query"
                test_case.passed = False
                test_case.error_message = "Query is empty"
                test_case.timestamp = datetime.now().isoformat()
                return test_case

            # Try to process the query
            config = get_config()
            confidence_threshold = config.reasoning.confidence_threshold if hasattr(config, 'reasoning') else 0.6

            response, metrics = rag_system.process_query(test_case.query, enable_evaluation=False)

            if response and response.answer:
                test_case.result = response.answer[:200]
                test_case.error_message = None

                # Check if system handled edge case gracefully
                test_case.passed = True
                if test_case.test_type == "ambiguous":
                    # Check if answer acknowledges ambiguity
                    has_clarification = any(word in response.answer.lower()
                        for word in ["clarify", "unclear", "multiple", "could mean"])
                    test_case.passed = has_clarification
                elif test_case.test_type == "no_answer":
                    # Check if system admits it doesn't know
                    has_admission = any(word in response.answer.lower()
                        for word in ["don't know", "don't have", "unclear", "no information"])
                    test_case.passed = has_admission or response.confidence_score < confidence_threshold
            else:
                test_case.result = "No answer generated"
                test_case.passed = False

            test_case.timestamp = datetime.now().isoformat()
            return test_case

        except Exception as e:
            test_case.result = "ERROR"
            test_case.passed = False
            test_case.error_message = str(e)
            test_case.timestamp = datetime.now().isoformat()
            logger.error(f"❌ Test {test_case.test_id} failed: {str(e)}")
            return test_case

    @staticmethod
    def run_all_tests(rag_system: "RAGSystem") -> List[AdversarialTestCase]:
        """Run all adversarial tests"""
        test_cases = AdversarialTestSuite.generate_test_cases()
        results = []

        for test_case in test_cases:
            result = AdversarialTestSuite.run_test_case(rag_system, test_case)
            results.append(result)

        return results
