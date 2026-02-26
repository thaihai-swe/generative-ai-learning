"""Configuration management for RAG system"""
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM Configuration"""
    api_key: str
    api_base_url: str
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 1000


@dataclass
class SearchConfig:
    """Search Configuration"""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    max_results: int = 3
    embedding_cache_size: int = 1000


@dataclass
class EvaluationConfig:
    """Evaluation Configuration"""
    confidence_threshold: float = 0.6
    check_facts: bool = False
    compute_ragas: bool = True
    max_facts_to_check: int = 5


@dataclass
class ReasoningConfig:
    """Reasoning Configuration"""
    query_expansion_count: int = 4
    multi_hop_steps: int = 3
    run_adversarial_tests: bool = False


@dataclass
class RAGConfig:
    """Complete RAG System Configuration"""
    llm: LLMConfig
    search: SearchConfig
    evaluation: EvaluationConfig
    reasoning: ReasoningConfig
    data_dir: str = "./json_data"
    db_path: str = "./chroma_db"

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load configuration from environment variables"""
        return cls(
            llm=LLMConfig(
                api_key=os.getenv("OPEN_AI_API_KEY", "lm-studio"),
                api_base_url=os.getenv(
                    "OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1"
                ),
                model_name=os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct"),
            ),
            search=SearchConfig(),
            evaluation=EvaluationConfig(),
            reasoning=ReasoningConfig(),
        )


# Global config instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get or initialize global configuration"""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set global configuration"""
    global _config
    _config = config
