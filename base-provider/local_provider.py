"""
Local LM Provider Implementation
For using local language models (via OpenAI-compatible endpoints)
Examples: Ollama, LocalAI, vLLM, Text Generation WebUI
"""

from typing import List, Optional

from .openai_provider import OpenAIProvider
from .base import Message, Response


class LocalLMProvider(OpenAIProvider):
    """
    Local LM provider implementation

    Inherits from OpenAIProvider to reuse OpenAI-compatible API logic.
    Designed for local models served via OpenAI-compatible endpoints.

    Examples:
        # Ollama (running at localhost:11434)
        provider = LocalLMProvider(
            model="llama2",
            api_base_url="http://localhost:11434/v1"
        )

        # LocalAI
        provider = LocalLMProvider(
            model="gpt4all-j",
            api_base_url="http://localhost:8080/v1"
        )

        # vLLM
        provider = LocalLMProvider(
            model="meta-llama/Llama-2-7b-chat-hf",
            api_base_url="http://localhost:8000/v1"
        )

        # Custom endpoint
        provider = LocalLMProvider(
            model="custom-model",
            api_base_url="http://127.0.0.1:5000/v1"
        )
    """

    def __init__(
        self,
        model: str,
        api_base_url: str = "http://localhost:11434/v1",
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize Local LM provider

        Args:
            model: Model name
            api_base_url: Base URL for the local server (default: Ollama)
            api_key: API key (optional for most local setups)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Additional arguments
        """
        super().__init__(
            model=model,
            api_key=api_key,
            api_base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"LocalLMProvider(model={self.model}, url={self.api_base_url})"


# Preset configurations for common local LM setups

class OllamaProvider(LocalLMProvider):
    """
    Ollama provider preset

    Usage:
        provider = OllamaProvider(model="llama2")
        provider.initialize()
        response = provider.complete([Message("user", "Hello")])
    """

    def __init__(self, model: str, host: str = "localhost", port: int = 11434, **kwargs):
        """
        Initialize Ollama provider

        Args:
            model: Model name (e.g., "llama2", "neural-chat", "mistral")
            host: Ollama host
            port: Ollama port
            **kwargs: Additional arguments
        """
        api_base_url = f"http://{host}:{port}/v1"
        super().__init__(model=model, api_base_url=api_base_url, **kwargs)


class LocalAIProvider(LocalLMProvider):
    """
    LocalAI provider preset

    Usage:
        provider = LocalAIProvider(model="gpt4all-j")
        provider.initialize()
        response = provider.complete([Message("user", "Hello")])
    """

    def __init__(self, model: str, host: str = "localhost", port: int = 8080, **kwargs):
        """
        Initialize LocalAI provider

        Args:
            model: Model name
            host: LocalAI host
            port: LocalAI port
            **kwargs: Additional arguments
        """
        api_base_url = f"http://{host}:{port}/v1"
        super().__init__(model=model, api_base_url=api_base_url, **kwargs)


class VLLMProvider(LocalLMProvider):
    """
    vLLM provider preset

    Usage:
        provider = VLLMProvider(model="meta-llama/Llama-2-7b-chat-hf")
        provider.initialize()
        response = provider.complete([Message("user", "Hello")])
    """

    def __init__(self, model: str, host: str = "localhost", port: int = 8000, **kwargs):
        """
        Initialize vLLM provider

        Args:
            model: Model name
            host: vLLM host
            port: vLLM port
            **kwargs: Additional arguments
        """
        api_base_url = f"http://{host}:{port}/v1"
        super().__init__(model=model, api_base_url=api_base_url, **kwargs)
