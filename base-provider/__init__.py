"""
Base Provider - A flexible LLM provider abstraction layer
Supports multiple providers: OpenAI, Local LMs, and more
"""

from .base import LLMProvider, Message, Response
from .config import ProviderConfig, load_config_from_env
from .errors import ProviderError, ConfigurationError, APIError
from .openai_provider import OpenAIProvider
from .local_provider import LocalLMProvider

__version__ = "1.0.0"
__all__ = [
    "LLMProvider",
    "Message",
    "Response",
    "ProviderConfig",
    "load_config_from_env",
    "ProviderError",
    "ConfigurationError",
    "APIError",
    "OpenAIProvider",
    "LocalLMProvider",
]
