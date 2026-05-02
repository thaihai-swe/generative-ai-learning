"""
Base Provider Abstract Class
Defines the interface all LLM providers must implement
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    """Message role enum"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Represents a single message in a conversation"""
    role: Role
    content: str

    def __init__(self, role: str, content: str):
        """
        Initialize a message

        Args:
            role: "system", "user", or "assistant"
            content: The message content
        """
        if isinstance(role, str):
            self.role = Role(role)
        else:
            self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API calls"""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class Response:
    """Represents a response from the LLM"""
    content: str
    model: str
    tokens_used: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None  # Store raw API response for debugging

    def __str__(self) -> str:
        return self.content

    def get_token_count(self) -> Optional[int]:
        """Get total tokens used"""
        if self.tokens_used:
            return sum(self.tokens_used.values())
        return None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers should inherit from this class and implement
    the abstract methods.
    """

    def __init__(self, model: str, **kwargs):
        """
        Initialize the provider

        Args:
            model: Model name/identifier
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.config = kwargs

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the provider (connect, authenticate, etc.)
        Should be called before using the provider
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection/cleanup resources"""
        pass

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Response:
        """
        Generate a completion

        Args:
            messages: List of Message objects
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            Response object
        """
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate a streaming completion

        Args:
            messages: List of Message objects
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Yields:
            Response objects (streamed chunks)
        """
        pass

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
