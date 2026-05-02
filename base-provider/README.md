# Base Provider

A flexible, production-ready LLM provider abstraction layer that supports multiple providers (OpenAI, Local LMs, and more).

**Key Features:**
- 🔄 **Multi-Provider Support**: OpenAI, Local LMs (Ollama, LocalAI, vLLM), and easily extensible
- 🎯 **Unified Interface**: Single API for all providers
- ⚙️ **Configuration Management**: Load from environment variables or code
- 🔌 **OpenAI-Compatible**: Works with any OpenAI-compatible API endpoint
- 📦 **Production-Ready**: Error handling, retries, logging, streaming
- 🛡️ **Type-Safe**: Full type hints for IDE support
- 📚 **Well-Documented**: Comprehensive examples and docstrings

## Installation

```bash
# Copy base-provider to your project or install from path
pip install -r base-provider/requirements.txt

# Or install as editable package
pip install -e base-provider/
```

## Quick Start

### 1. OpenAI API

```python
from base_provider import OpenAIProvider, Message

provider = OpenAIProvider(
    model="gpt-3.5-turbo",
    api_key="sk-...",
)

provider.initialize()
response = provider.complete([
    Message("user", "Hello!")
])
print(response.content)
provider.close()
```

### 2. Local LM (Ollama)

```python
from base_provider import OllamaProvider, Message

provider = OllamaProvider(model="llama2")
provider.initialize()

response = provider.complete([
    Message("user", "What is Python?")
])
print(response.content)
provider.close()
```

### 3. Custom Endpoint

```python
from base_provider import OpenAIProvider, Message

# Works with any OpenAI-compatible endpoint
provider = OpenAIProvider(
    model="meta-llama-3.1-8b-instruct",
    api_key="not-needed",
    api_base_url="http://127.0.0.1:1234/v1"
)

provider.initialize()
response = provider.complete([Message("user", "Hello")])
print(response.content)
provider.close()
```

### 4. Context Manager (Auto Cleanup)

```python
from base_provider import OpenAIProvider, Message

with OpenAIProvider(
    model="gpt-3.5-turbo",
    api_key="sk-...",
) as provider:
    response = provider.complete([Message("user", "Hello")])
    print(response.content)
# Provider automatically closed
```

### 5. Configuration from Environment

```python
import os
from base_provider import load_config_from_env, OpenAIProvider

# Set environment variables:
# PROVIDER_TYPE=openai
# MODEL=gpt-4
# API_KEY=sk-...
# API_BASE_URL=https://api.openai.com/v1

config = load_config_from_env()
provider = OpenAIProvider(**config.to_provider_kwargs())
provider.initialize()
```

## Configuration

### Environment Variables

Use environment variables to configure the provider without code changes:

```bash
# Basic configuration
PROVIDER_TYPE=openai
MODEL=gpt-3.5-turbo
API_KEY=sk-...
API_BASE_URL=https://api.openai.com/v1

# Optional settings
TEMPERATURE=0.7
MAX_TOKENS=1000
TIMEOUT=30
MAX_RETRIES=3
ORGANIZATION=org-123  # For OpenAI organization
```

### With Custom Prefix

```bash
# Set environment variables with prefix
BOT_PROVIDER_TYPE=openai
BOT_MODEL=gpt-4
BOT_API_KEY=sk-...
BOT_API_BASE_URL=https://api.openai.com/v1

# Load with prefix
config = load_config_from_env(prefix="BOT")
```

### From Code

```python
from base_provider import ProviderConfig, OpenAIProvider

config = ProviderConfig(
    provider_type="openai",
    model="gpt-4",
    api_key="sk-...",
    api_base_url="https://api.openai.com/v1",
    temperature=0.7,
    max_tokens=1000,
)

provider = OpenAIProvider(**config.to_provider_kwargs())
provider.initialize()
```

## Supported Providers

### OpenAI (Official API)

```python
from base_provider import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4",  # or gpt-3.5-turbo
    api_key="sk-...",
)
```

**Supported Models**: GPT-4, GPT-4-Turbo, GPT-3.5-Turbo, etc.

### Ollama (Local)

```python
from base_provider import OllamaProvider

provider = OllamaProvider(
    model="llama2",  # or neural-chat, mistral, etc.
    host="localhost",
    port=11434,
)
```

**Setup**:
```bash
# Install Ollama from https://ollama.ai
# Run Ollama
ollama serve

# In another terminal, pull a model
ollama pull llama2
```

### LocalAI

```python
from base_provider import LocalAIProvider

provider = LocalAIProvider(
    model="gpt4all-j",
    host="localhost",
    port=8080,
)
```

### vLLM

```python
from base_provider import VLLMProvider

provider = VLLMProvider(
    model="meta-llama/Llama-2-7b-chat-hf",
    host="localhost",
    port=8000,
)
```

### Custom Endpoint

```python
from base_provider import OpenAIProvider

provider = OpenAIProvider(
    model="any-model-name",
    api_key="your-key",
    api_base_url="http://your-endpoint/v1"
)
```

## API Reference

### OpenAIProvider

```python
class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize OpenAI provider"""

    def initialize(self) -> None:
        """Initialize the client"""

    def close(self) -> None:
        """Close the client"""

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Response:
        """Generate a completion"""

    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate a streaming completion"""
```

### Message

```python
from base_provider import Message

msg = Message(
    role="user",  # "system", "user", or "assistant"
    content="What is Python?"
)

# Or with Role enum
from base_provider.base import Role
msg = Message(role=Role.USER, content="Hello")
```

### Response

```python
response = provider.complete(messages)
print(response.content)          # The generated text
print(response.model)            # Model used
print(response.tokens_used)      # {"prompt": 10, "completion": 20, "total": 30}
print(response.finish_reason)    # "stop", "length", etc.
print(response.raw_response)     # Raw API response object
```

### Configuration Classes

```python
from base_provider import ProviderConfig, load_config_from_env

# From environment
config = load_config_from_env(prefix="BOT")

# From dictionary
config = ProviderConfig(
    provider_type="openai",
    model="gpt-4",
    api_key="sk-...",
)

# Convert back
config.to_dict()                 # Get all settings as dict
config.to_provider_kwargs()      # Get kwargs for provider init
```

## Error Handling

```python
from base_provider import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ConfigurationError,
)

try:
    response = provider.complete(messages)
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except RateLimitError as e:
    print(f"Rate limit: {e}")
except APIError as e:
    print(f"API error: {e}")
except ConfigurationError as e:
    print(f"Config error: {e}")
```

## Streaming

```python
# Get streamed responses
for chunk in provider.stream(messages, max_tokens=200):
    print(chunk.content, end="", flush=True)
print()
```

## Integration Pattern

Recommended way to integrate into your project:

```python
# config.py
import os
from base_provider import load_config_from_env, OpenAIProvider

def get_llm_provider():
    """Get or create the LLM provider (singleton pattern)"""
    if not hasattr(get_llm_provider, "_provider"):
        config = load_config_from_env(prefix="APP")
        provider = OpenAIProvider(**config.to_provider_kwargs())
        provider.initialize()
        get_llm_provider._provider = provider

    return get_llm_provider._provider

# In your application code
from config import get_llm_provider
from base_provider import Message

provider = get_llm_provider()
response = provider.complete([
    Message("system", "You are a helpful assistant"),
    Message("user", "Hello!")
])
print(response.content)
```

## File Structure

```
base-provider/
├── __init__.py              # Package exports
├── base.py                  # Base provider class
├── config.py                # Configuration management
├── errors.py                # Custom exceptions
├── openai_provider.py       # OpenAI implementation
├── local_provider.py        # Local LM implementations
├── examples.py              # Usage examples
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Extending the Provider

To add support for a new provider:

1. Inherit from `LLMProvider` base class
2. Implement required methods: `initialize()`, `close()`, `complete()`, `stream()`
3. Use the `Message` and `Response` classes for compatibility

Example:

```python
from base_provider import LLMProvider, Message, Response
from typing import List, Optional

class MyProvider(LLMProvider):
    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.client = None

    def initialize(self) -> None:
        # Initialize your client here
        self.client = MyLLMClient()

    def close(self) -> None:
        # Cleanup
        if self.client:
            self.client.close()

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Response:
        # Convert messages to your format
        msgs = [msg.to_dict() for msg in messages]

        # Call your API
        result = self.client.generate(
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Return Response object
        return Response(
            content=result["text"],
            model=self.model,
        )

    def stream(self, messages, temperature=0.7, max_tokens=None, **kwargs):
        # Similar to complete but yield chunks
        pass
```

## Logging

Enable logging to see what's happening:

```python
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("base_provider")

# Now you'll see debug info about API calls
provider = OpenAIProvider(...)
provider.initialize()
response = provider.complete(messages)  # See debug output
```

## Testing

```python
from unittest.mock import Mock, patch
from base_provider import OpenAIProvider, Message

def test_openai_provider():
    # Mock the OpenAI client
    with patch("openai.OpenAI") as mock_client:
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 10
        mock_client.return_value.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="test", api_key="test")
        provider.initialize()

        response = provider.complete([Message("user", "test")])
        assert response.content == "Test response"
```

## Performance Tips

1. **Reuse provider instance**: Don't create new providers for each request
2. **Use streaming for long responses**: Reduces memory usage
3. **Set appropriate timeouts**: Match your use case
4. **Implement retry logic**: For production systems
5. **Monitor token usage**: Track costs and performance
6. **Cache responses**: When appropriate

## Contributing

To add support for new providers:

1. Create a new file: `provider_name.py`
2. Implement the `LLMProvider` interface
3. Add preset configurations if applicable
4. Add examples in `examples.py`
5. Update documentation

## License

This is part of the Generative AI Learning series.

## Support

- See `examples.py` for usage examples
- Check docstrings in each module for detailed API docs
- Review error handling for common issues
