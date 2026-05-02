"""
OpenAI Provider Implementation
Supports both official OpenAI API and compatible APIs (local LMs via OpenAI-compatible endpoints)
"""

from typing import List, Optional, Dict, Any
import logging

from openai import OpenAI, APIError as OpenAIAPIError, APIConnectionError, RateLimitError as OpenAIRateLimitError
from openai import AuthenticationError as OpenAIAuthenticationError

from .base import LLMProvider, Message, Response
from .errors import APIError, ConfigurationError, RateLimitError, AuthenticationError, StreamingError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation

    Supports:
    - Official OpenAI API (GPT-4, GPT-3.5, etc.)
    - Local LMs with OpenAI-compatible API (Ollama, LocalAI, etc.)
    - Any OpenAI-compatible endpoint

    Examples:
        # Official OpenAI
        provider = OpenAIProvider(
            model="gpt-4",
            api_key="sk-...",
        )

        # Local LM via Ollama
        provider = OpenAIProvider(
            model="llama2",
            api_key="dummy",  # LocalAI doesn't need real key
            api_base_url="http://localhost:11434/v1"
        )

        # Custom endpoint
        provider = OpenAIProvider(
            model="meta-llama-3.1-8b-instruct",
            api_key="your-key",
            api_base_url="http://127.0.0.1:1234/v1"
        )
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize OpenAI provider

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "llama2")
            api_key: API key for authentication
            api_base_url: Base URL for the API
            organization: Organization ID (for OpenAI)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Additional arguments
        """
        super().__init__(model=model, **kwargs)

        self.api_key = api_key
        self.api_base_url = api_base_url
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize OpenAI client"""
        if self._initialized:
            return

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self._initialized = True
            logger.info(f"OpenAI provider initialized with model: {self.model}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI provider: {e}")

    def close(self) -> None:
        """Close the OpenAI client"""
        if self.client:
            self.client.close()
            self.client = None
            self._initialized = False

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Response:
        """
        Generate a completion using OpenAI API

        Args:
            messages: List of Message objects
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for OpenAI API

        Returns:
            Response object

        Raises:
            ConfigurationError: If not initialized
            RateLimitError: If rate limit exceeded
            AuthenticationError: If authentication fails
            APIError: If API call fails
        """
        if not self._initialized:
            raise ConfigurationError("Provider not initialized. Call initialize() first.")

        try:
            # Build message list
            message_dicts = [msg.to_dict() for msg in messages]

            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": message_dicts,
                "temperature": temperature,
            }

            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # Add any additional kwargs
            request_params.update(kwargs)

            logger.debug(f"Calling OpenAI API with model: {self.model}")

            # Make API call
            response = self.client.chat.completions.create(**request_params)

            # Extract response
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Extract token usage
            tokens_used = None
            if hasattr(response, 'usage') and response.usage:
                tokens_used = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                }

            logger.info(f"Received response from {self.model}, tokens: {tokens_used}")

            return Response(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=response
            )

        except OpenAIRateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise RateLimitError(f"Rate limit exceeded: {e}", status_code=429)

        except OpenAIAuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(f"Authentication failed: {e}", status_code=401)

        except OpenAIAPIError as e:
            logger.error(f"API error: {e}")
            status_code = getattr(e, 'status_code', None)
            raise APIError(f"API error: {e}", status_code=status_code)

        except APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise APIError(f"Connection error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(f"Unexpected error: {e}")

    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate a streaming completion using OpenAI API

        Args:
            messages: List of Message objects
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for OpenAI API

        Yields:
            Response objects (streamed chunks)

        Raises:
            ConfigurationError: If not initialized
            StreamingError: If streaming fails
            APIError: If API call fails
        """
        if not self._initialized:
            raise ConfigurationError("Provider not initialized. Call initialize() first.")

        try:
            # Build message list
            message_dicts = [msg.to_dict() for msg in messages]

            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": message_dicts,
                "temperature": temperature,
                "stream": True,  # Enable streaming
            }

            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # Add any additional kwargs
            request_params.update(kwargs)

            logger.debug(f"Calling OpenAI API (streaming) with model: {self.model}")

            # Make streaming API call
            with self.client.chat.completions.create(**request_params) as stream:
                accumulated_content = ""

                for chunk in stream:
                    # Extract delta content
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        accumulated_content += content

                        # Yield chunk response
                        yield Response(
                            content=content,
                            model=self.model,
                            finish_reason=chunk.choices[0].finish_reason,
                            raw_response=chunk
                        )

                logger.info(f"Streaming complete, total content length: {len(accumulated_content)}")

        except OpenAIRateLimitError as e:
            logger.error(f"Rate limit exceeded during streaming: {e}")
            raise RateLimitError(f"Rate limit exceeded: {e}", status_code=429)

        except OpenAIAuthenticationError as e:
            logger.error(f"Authentication failed during streaming: {e}")
            raise AuthenticationError(f"Authentication failed: {e}", status_code=401)

        except OpenAIAPIError as e:
            logger.error(f"API error during streaming: {e}")
            raise StreamingError(f"API error: {e}")

        except APIConnectionError as e:
            logger.error(f"Connection error during streaming: {e}")
            raise StreamingError(f"Connection error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}")
            raise StreamingError(f"Unexpected error: {e}")
