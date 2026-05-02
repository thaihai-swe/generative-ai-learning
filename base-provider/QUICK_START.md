# Base Provider - Quick Start

Get up and running with the Base Provider in 5 minutes.

## Installation

```bash
# Navigate to base-provider directory
cd base-provider

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as editable package
pip install -e .
```

## Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# For OpenAI: Add your API key
# For Local LM: Set API_BASE_URL to your server
```

## Example 1: OpenAI (30 seconds)

```python
from base_provider import OpenAIProvider, Message

with OpenAIProvider(
    model="gpt-3.5-turbo",
    api_key="sk-your-api-key",
) as provider:
    response = provider.complete([
        Message("user", "What is Python?")
    ])
    print(response.content)
```

## Example 2: Local LM - Ollama (30 seconds)

```python
# Make sure Ollama is running: ollama serve

from base_provider import OllamaProvider, Message

with OllamaProvider(model="llama2") as provider:
    response = provider.complete([
        Message("user", "Write a haiku about coding")
    ])
    print(response.content)
```

## Example 3: Any OpenAI-Compatible Endpoint (30 seconds)

```python
from base_provider import OpenAIProvider, Message

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

## Example 4: Load from Environment

```python
import os
from dotenv import load_dotenv
from base_provider import load_config_from_env, OpenAIProvider, Message

load_dotenv()

# Load config from .env
config = load_config_from_env()

# Create and use provider
with OpenAIProvider(**config.to_provider_kwargs()) as provider:
    response = provider.complete([
        Message("user", "Hello!")
    ])
    print(response.content)
```

## Run Examples

```bash
# Run all examples
python examples.py

# Run specific example (uncomment in examples.py)
python -c "from examples import example_openai_official; example_openai_official()"
```

## Common Setups

### OpenAI Official API

```bash
# 1. Get API key from https://platform.openai.com/api-keys
# 2. Set in .env:
PROVIDER_TYPE=openai
MODEL=gpt-3.5-turbo
API_KEY=sk-your-key
API_BASE_URL=https://api.openai.com/v1
```

### Ollama (Local)

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Start Ollama: ollama serve
# 3. Pull model: ollama pull llama2
# 4. Use in Python:

from base_provider import OllamaProvider, Message

with OllamaProvider(model="llama2") as provider:
    response = provider.complete([Message("user", "Hello")])
    print(response.content)
```

### LocalAI

```bash
# 1. Follow LocalAI setup at https://localai.io
# 2. Set in .env:
PROVIDER_TYPE=openai
MODEL=gpt4all-j
API_KEY=not-needed
API_BASE_URL=http://localhost:8080/v1
```

### vLLM

```bash
# 1. Install vLLM: pip install vllm
# 2. Start server:
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-2-7b-chat-hf
# 3. Use in .env:
PROVIDER_TYPE=openai
MODEL=meta-llama/Llama-2-7b-chat-hf
API_KEY=not-needed
API_BASE_URL=http://localhost:8000/v1
```

## Typical Usage Pattern

```python
# config.py
from base_provider import load_config_from_env, OpenAIProvider

def get_llm():
    """Get the LLM provider"""
    config = load_config_from_env()
    provider = OpenAIProvider(**config.to_provider_kwargs())
    provider.initialize()
    return provider

# app.py
from base_provider import Message
from config import get_llm

llm = get_llm()
response = llm.complete([
    Message("system", "You are a helpful assistant"),
    Message("user", "What is machine learning?"),
])
print(response.content)
llm.close()
```

## Streaming

```python
from base_provider import OpenAIProvider, Message

with OpenAIProvider(
    model="gpt-3.5-turbo",
    api_key="sk-...",
) as provider:
    for chunk in provider.stream([
        Message("user", "Write a story about an AI")
    ]):
        print(chunk.content, end="", flush=True)
```

## Error Handling

```python
from base_provider import (
    OpenAIProvider,
    Message,
    AuthenticationError,
    RateLimitError,
    APIError,
)

try:
    provider = OpenAIProvider(model="gpt-4", api_key="invalid")
    provider.initialize()
    response = provider.complete([Message("user", "Hi")])
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except APIError as e:
    print(f"API error: {e}")
```

## Next Steps

1. Read [README.md](README.md) for full documentation
2. Check [examples.py](examples.py) for more examples
3. Explore the code in:
   - `base.py` - Base provider class
   - `openai_provider.py` - OpenAI implementation
   - `local_provider.py` - Local LM implementations
4. Create your own provider by inheriting from `LLMProvider`

## Troubleshooting

**"ModuleNotFoundError: No module named 'base_provider'"**
- Make sure you're in the right directory or installed with `pip install -e .`

**"API Key is invalid"**
- Check your API key in .env
- Make sure it's not expired

**"Connection to API failed"**
- Check your internet connection
- Verify API_BASE_URL is correct
- For local LMs, make sure the server is running

**"Model not found"**
- Verify the model name is correct
- For OpenAI, check if the model is available in your account
- For local LMs, pull the model first (e.g., `ollama pull llama2`)

## Integration with Your Project

Copy `base-provider/` to your project:

```
your-project/
├── base-provider/          # Shared LLM provider
├── your-app/
│   ├── config.py
│   └── main.py
└── requirements.txt
```

Then in your code:

```python
from base_provider import OpenAIProvider, Message

provider = OpenAIProvider(...)
response = provider.complete([Message("user", "Hello")])
```

## API Summary

```python
# Import
from base_provider import (
    OpenAIProvider,           # OpenAI implementation
    OllamaProvider,           # Ollama preset
    LocalAIProvider,          # LocalAI preset
    VLLMProvider,             # vLLM preset
    Message,                  # Message class
    Response,                 # Response class
    ProviderConfig,           # Config class
    load_config_from_env,     # Load config from env
)

# Create provider
provider = OpenAIProvider(model="gpt-4", api_key="sk-...")

# Initialize (or use context manager)
provider.initialize()

# Send message
response = provider.complete([
    Message("user", "Hello!"),
])

# Get response
print(response.content)        # The generated text
print(response.tokens_used)    # Token usage
print(response.model)          # Model used

# Stream response
for chunk in provider.stream([Message("user", "Write a story")]):
    print(chunk.content, end="")

# Cleanup
provider.close()
```

---

Happy coding! 🚀
