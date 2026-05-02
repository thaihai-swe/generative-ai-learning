# Base Provider - Documentation Index

A comprehensive index of all base-provider documentation and resources.

## 📋 Quick Navigation

### 👤 For Different Users

**I'm new to this project**
- Start here: [QUICK_START.md](QUICK_START.md) - 5 minute setup
- Then: [README.md](README.md#quick-start) - Feature overview
- Finally: [examples.py](examples.py) - Real code examples

**I'm an experienced developer**
- Jump to: [README.md](README.md) - Full API reference
- Check: [examples.py](examples.py#example-8-integration-pattern) - Integration patterns
- Read: [base.py](base.py) - Architecture and design

**I want to extend this**
- Read: [README.md](README.md#extending-the-provider) - Extension guide
- Study: [openai_provider.py](openai_provider.py) - Implementation reference
- Check: [test_provider.py](test_provider.py) - Testing patterns

**I want to test my code**
- See: [test_provider.py](test_provider.py) - Test examples
- Read: [README.md](README.md#testing) - Testing guide
- Example: [examples.py](examples.py#example-7-error-handling) - Error testing

## 📚 Documentation Files

### QUICK_START.md
**What**: 5-minute setup and common examples
**Length**: ~220 lines
**Contains**:
- Installation steps
- 5 quick examples (OpenAI, Ollama, custom, env, streaming)
- Common provider setups
- Integration patterns
- Troubleshooting guide

**When to use**: First time using base-provider

### README.md
**What**: Comprehensive documentation
**Length**: ~350 lines
**Contains**:
- Full feature overview
- Installation and setup
- Quick start examples (5 code blocks)
- Configuration guide
- Supported providers section
- API reference (all classes and methods)
- Error handling guide
- Streaming examples
- Integration patterns
- File structure
- Extension guide (for custom providers)
- Testing guide
- Performance tips

**When to use**: Complete reference, API questions, extending

### SKILL.md (This File)
**What**: Documentation index and navigation guide
**Length**: ~260 lines
**Contains**:
- File navigation guide
- Content summary for each file
- Use cases for different users
- API quick reference
- Common tasks and how to do them

**When to use**: Finding what you need

## 💻 Code Files

### __init__.py
**What**: Package initialization
**Length**: ~50 lines
**Contains**:
- Package version
- Public API exports
- All classes and functions available at top level

**Key exports**:
- `LLMProvider`, `Message`, `Response`
- `OpenAIProvider`, `LocalLMProvider`, `OllamaProvider`, etc.
- `ProviderConfig`, `load_config_from_env`
- All error classes

**When to use**: Understanding public API

### base.py
**What**: Abstract base classes and data structures
**Length**: ~150 lines
**Contains**:
- `Role` enum (system, user, assistant)
- `Message` class (role + content)
- `Response` class (content, model, tokens, etc.)
- `LLMProvider` abstract base class
- Abstract methods: initialize(), close(), complete(), stream()

**Key concepts**:
- All providers inherit from LLMProvider
- All providers use Message/Response for compatibility
- Context manager support (__enter__/__exit__)

**When to use**: Understanding provider interface, extending

### config.py
**What**: Configuration management
**Length**: ~200 lines
**Contains**:
- `ProviderType` enum
- `ProviderConfig` dataclass
- `load_config_from_env()` function
- `load_config_from_dict()` function

**Key features**:
- Load configuration from environment variables
- Support custom prefixes (e.g., "BOT_", "APP_")
- Validation built-in
- Convert to provider kwargs

**When to use**: Loading configuration, setting up providers

### errors.py
**What**: Custom exception hierarchy
**Length**: ~40 lines
**Contains**:
- `ProviderError` (base exception)
- `ConfigurationError` (config issues)
- `APIError` (API request failures)
- `RateLimitError` (rate limiting)
- `AuthenticationError` (auth issues)
- `ModelNotFoundError` (unknown model)
- `StreamingError` (streaming issues)

**Key design**:
- All inherit from ProviderError
- APIError stores status_code and raw response
- Use for better error handling

**When to use**: Error handling in your code

### openai_provider.py
**What**: OpenAI API implementation
**Length**: ~350 lines
**Contains**:
- `OpenAIProvider` class (full implementation)
- Support for OpenAI official API
- Support for any OpenAI-compatible endpoint
- Error handling and conversion
- Token tracking
- Logging

**Key features**:
- Works with official OpenAI API
- Works with local LMs via compatible endpoints
- Customizable timeouts and retries
- Streaming support
- Full error handling

**When to use**: Understanding OpenAI implementation, extending

### local_provider.py
**What**: Local LM provider implementations
**Length**: ~120 lines
**Contains**:
- `LocalLMProvider` (base for local LMs)
- `OllamaProvider` (Ollama preset)
- `LocalAIProvider` (LocalAI preset)
- `VLLMProvider` (vLLM preset)

**Key design**:
- All inherit from OpenAIProvider (reuse OpenAI-compatible logic)
- Presets provide sensible defaults
- Easy to set up

**When to use**: Using local LMs, creating presets

### examples.py
**What**: 8 comprehensive usage examples
**Length**: ~380 lines
**Contains**:
1. OpenAI official API usage
2. Context manager pattern
3. Environment variable configuration
4. Custom endpoint (LM Studio)
5. Ollama provider preset
6. Streaming responses
7. Error handling patterns
8. Integration pattern (recommended)

**When to use**: Learning how to use, copy-paste patterns

### test_provider.py
**What**: Testing examples and patterns
**Length**: ~300 lines
**Contains**:
- Unit tests for Message class
- Unit tests for Response class
- Unit tests for ProviderConfig
- Unit tests for OpenAIProvider
- Error handling tests
- Mocking examples

**When to use**: Writing your own tests, understanding testing patterns

## 🔧 Configuration Files

### requirements.txt
**Contains**:
```
openai>=1.0.0
python-dotenv>=1.0.0
requests>=2.31.0
```

**When to use**: Installing dependencies

### .env.example
**Contains**:
- Example environment variables
- Multiple provider configurations
- Prefix examples

**When to use**: Setting up your .env file

### setup.py
**Contains**:
- Package metadata
- Installation configuration
- Dependencies list
- Classifiers

**When to use**: Installing as a package with `pip install -e .`

### .gitignore
**Contains**:
- Ignore patterns for Git
- Python-specific ignores
- IDE ignores
- Environment ignores

**When to use**: Already configured for Git

## 📖 Common Tasks

### Task 1: Set Up Base Provider

1. Navigate to base-provider directory
2. Create virtual environment: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Copy .env: `cp .env.example .env`
6. Edit .env with your API keys

**Reference**: [QUICK_START.md](QUICK_START.md#installation)

### Task 2: Use OpenAI API

```python
from base_provider import OpenAIProvider, Message

with OpenAIProvider(
    model="gpt-3.5-turbo",
    api_key="sk-your-key",
) as provider:
    response = provider.complete([
        Message("user", "Hello!")
    ])
    print(response.content)
```

**Reference**: [QUICK_START.md](QUICK_START.md#example-1-openai-30-seconds)

### Task 3: Use Ollama (Local)

```python
from base_provider import OllamaProvider, Message

with OllamaProvider(model="llama2") as provider:
    response = provider.complete([
        Message("user", "What is Python?")
    ])
    print(response.content)
```

**Reference**: [QUICK_START.md](QUICK_START.md#example-2-local-lm---ollama-30-seconds)

### Task 4: Load From Environment

```python
from base_provider import load_config_from_env, OpenAIProvider

config = load_config_from_env(prefix="BOT")
with OpenAIProvider(**config.to_provider_kwargs()) as provider:
    response = provider.complete([Message("user", "Hello")])
```

**Reference**: [QUICK_START.md](QUICK_START.md#example-4-load-from-environment)

### Task 5: Handle Errors

```python
from base_provider import (
    OpenAIProvider, Message,
    AuthenticationError, RateLimitError, APIError
)

try:
    provider = OpenAIProvider(model="gpt-4", api_key="sk-...")
    provider.initialize()
    response = provider.complete([Message("user", "Hi")])
except AuthenticationError:
    print("Authentication failed")
except RateLimitError:
    print("Rate limit exceeded")
except APIError as e:
    print(f"API error: {e}")
```

**Reference**: [examples.py](examples.py) (Example 7)

### Task 6: Stream Long Responses

```python
from base_provider import OpenAIProvider, Message

with OpenAIProvider(
    model="gpt-4",
    api_key="sk-...",
) as provider:
    for chunk in provider.stream([
        Message("user", "Write a long story")
    ]):
        print(chunk.content, end="", flush=True)
```

**Reference**: [examples.py](examples.py) (Example 6)

### Task 7: Create Custom Provider

1. Inherit from `LLMProvider`
2. Implement: `initialize()`, `close()`, `complete()`, `stream()`
3. Use `Message` and `Response` classes
4. Handle errors with custom exceptions

**Reference**: [README.md](README.md#extending-the-provider)

### Task 8: Test Your Code

```python
from unittest.mock import patch, Mock
from base_provider import OpenAIProvider, Message

@patch("openai.OpenAI")
def test_complete(mock_openai):
    mock_response = Mock()
    mock_response.choices[0].message.content = "Hello!"
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider(model="gpt-4", api_key="test")
    provider.initialize()
    response = provider.complete([Message("user", "Hi")])
    assert response.content == "Hello!"
```

**Reference**: [test_provider.py](test_provider.py)

### Task 9: Integrate Into Your Project

1. Create `config.py` with singleton:
   ```python
   from base_provider import load_config_from_env, OpenAIProvider

   def get_llm_provider():
       config = load_config_from_env(prefix="YOUR_APP")
       provider = OpenAIProvider(**config.to_provider_kwargs())
       provider.initialize()
       return provider
   ```

2. Use in your code:
   ```python
   from config import get_llm_provider
   from base_provider import Message

   provider = get_llm_provider()
   response = provider.complete([Message("user", "Hello")])
   ```

**Reference**: [examples.py](examples.py) (Example 8)

### Task 10: Switch Providers

Just change environment variables or configuration - no code changes needed:

```bash
# Switch from OpenAI to Ollama
# BEFORE: PROVIDER_TYPE=openai MODEL=gpt-4 API_KEY=sk-...
# AFTER:  PROVIDER_TYPE=openai MODEL=llama2 API_BASE_URL=http://localhost:11434/v1
```

Same code works with all providers!

## 🎯 Key Classes and Methods

### LLMProvider (Abstract Base)
```python
class LLMProvider:
    def initialize(self) -> None: ...
    def close(self) -> None: ...
    def complete(messages: List[Message], ...) -> Response: ...
    def stream(messages: List[Message], ...): ...  # yields Response chunks
    def __enter__(self): ...
    def __exit__(self, *args): ...
```

### Message
```python
class Message:
    role: Role  # system, user, or assistant
    content: str

    def to_dict(self) -> dict: ...
```

### Response
```python
class Response:
    content: str
    model: str
    tokens_used: dict  # {"prompt": int, "completion": int, "total": int}
    finish_reason: str
    raw_response: any

    def get_token_count(self) -> int: ...
```

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
    ): ...
```

### ProviderConfig
```python
class ProviderConfig:
    provider_type: ProviderType
    model: str
    api_key: Optional[str]
    api_base_url: Optional[str]
    temperature: float
    max_tokens: Optional[int]

    def to_provider_kwargs(self) -> dict: ...
    def to_dict(self) -> dict: ...
```

### Configuration Functions
```python
def load_config_from_env(
    provider_type: Optional[str] = None,
    prefix: str = ""
) -> ProviderConfig: ...

def load_config_from_dict(config_dict: dict) -> ProviderConfig: ...
```

### Exception Classes
- `ProviderError` - Base exception
- `ConfigurationError` - Configuration issues
- `APIError` - API request failures
- `RateLimitError` - Rate limiting
- `AuthenticationError` - Authentication failures
- `ModelNotFoundError` - Model not found
- `StreamingError` - Streaming issues

## 🔗 File Relationships

```
__init__.py (exports)
    ↓
base.py (LLMProvider abstract class)
    ↓
    ├─→ openai_provider.py (OpenAI implementation)
    │       ↓
    │   local_provider.py (Local LM presets)
    │
    └─→ errors.py (Exception classes)

config.py (Configuration management)
    ↓
    Uses: LLMProvider, errors

examples.py (Usage examples)
    ↓
    Uses: All of the above

test_provider.py (Testing)
    ↓
    Tests: All modules
```

## 📊 Quick Statistics

| File | Lines | Purpose |
|------|-------|---------|
| __init__.py | 50 | Package initialization |
| base.py | 150 | Abstract base classes |
| config.py | 200 | Configuration management |
| errors.py | 40 | Exception hierarchy |
| openai_provider.py | 350 | OpenAI implementation |
| local_provider.py | 120 | Local LM implementations |
| examples.py | 380 | Usage examples |
| test_provider.py | 300 | Testing examples |
| README.md | 350+ | Full documentation |
| QUICK_START.md | 220 | Quick start guide |
| This file | 260 | Documentation index |
| **TOTAL** | **2,400+** | **Complete package** |

## 🚀 Getting Started Checklist

- [ ] Read [QUICK_START.md](QUICK_START.md)
- [ ] Set up virtual environment
- [ ] Install requirements
- [ ] Copy and edit .env file
- [ ] Run [examples.py](examples.py)
- [ ] Try your first API call
- [ ] Integrate into your project
- [ ] Write tests with [test_provider.py](test_provider.py)

## ❓ FAQ

**Q: Which file should I read first?**
A: Start with [QUICK_START.md](QUICK_START.md) - it's only 220 lines and covers setup and basic examples.

**Q: How do I switch from OpenAI to Ollama?**
A: Just change environment variables or configuration - the code stays the same.

**Q: Can I use this with multiple LLM providers in one app?**
A: Yes! Use prefixes in environment variables: `BOT_PROVIDER_TYPE`, `APP_PROVIDER_TYPE`, etc.

**Q: How do I add support for a new provider?**
A: Inherit from `LLMProvider` and implement the abstract methods. See [README.md](README.md#extending-the-provider).

**Q: Where are the tests?**
A: See [test_provider.py](test_provider.py) for examples and patterns.

**Q: Is this production-ready?**
A: Yes! It includes error handling, retries, logging, streaming, and comprehensive testing examples.

## 📞 Support

- **API Questions**: See [README.md](README.md#api-reference)
- **Setup Issues**: See [QUICK_START.md](QUICK_START.md#troubleshooting)
- **Code Examples**: See [examples.py](examples.py)
- **Architecture**: See [base.py](base.py) and [openai_provider.py](openai_provider.py)
- **Testing**: See [test_provider.py](test_provider.py)

---

**Version**: 1.0.0
**Last Updated**: Latest session
**Status**: Production-ready ✓
