"""
Examples of using the Base Provider
Demonstrates how to integrate and use the provider in your projects
"""

import os
from dotenv import load_dotenv

from base_provider import (
    OpenAIProvider,
    LocalLMProvider,
    OllamaProvider,
    Message,
    load_config_from_env,
    ConfigurationError,
)


# Load environment variables
load_dotenv()


# ============================================================================
# Example 1: Using OpenAI Provider (Official OpenAI API)
# ============================================================================

def example_openai_official():
    """Use official OpenAI API"""
    print("=" * 80)
    print("Example 1: OpenAI Official API")
    print("=" * 80)

    try:
        provider = OpenAIProvider(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        provider.initialize()

        messages = [
            Message("system", "You are a helpful AI assistant."),
            Message("user", "What is Python?"),
        ]

        response = provider.complete(messages, temperature=0.7)
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.tokens_used}")

        provider.close()

    except ConfigurationError as e:
        print(f"Configuration error: {e}")


# ============================================================================
# Example 2: Using Context Manager (Auto cleanup)
# ============================================================================

def example_with_context_manager():
    """Use provider with context manager for automatic cleanup"""
    print("\n" + "=" * 80)
    print("Example 2: Context Manager (Auto-cleanup)")
    print("=" * 80)

    try:
        with OpenAIProvider(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
        ) as provider:
            messages = [
                Message("user", "Write a short poem about Python"),
            ]

            response = provider.complete(messages, temperature=0.9)
            print(f"Response:\n{response.content}")

        # Provider is automatically closed here
        print("Provider closed automatically")

    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# Example 3: Loading Configuration from Environment
# ============================================================================

def example_load_from_env():
    """Load configuration from environment variables"""
    print("\n" + "=" * 80)
    print("Example 3: Load Configuration from Environment")
    print("=" * 80)

    # Set environment variables
    os.environ["PROVIDER_TYPE"] = "openai"
    os.environ["MODEL"] = "gpt-4"
    os.environ["API_KEY"] = "sk-your-key-here"

    try:
        config = load_config_from_env()
        print(f"Provider type: {config.provider_type}")
        print(f"Model: {config.model}")
        print(f"API Base URL: {config.api_base_url}")

        # Create provider from config
        provider = OpenAIProvider(**config.to_provider_kwargs())
        print(f"Provider initialized: {provider}")

    except ConfigurationError as e:
        print(f"Configuration error: {e}")


# ============================================================================
# Example 4: Using Custom API Endpoint (e.g., Local LM)
# ============================================================================

def example_local_lm():
    """Use local LM with custom endpoint"""
    print("\n" + "=" * 80)
    print("Example 4: Local LM (Custom Endpoint)")
    print("=" * 80)

    # This assumes you have a local LM server running
    # Example: http://127.0.0.1:1234/v1 (could be Ollama, LocalAI, etc.)

    try:
        provider = OpenAIProvider(
            model="meta-llama-3.1-8b-instruct",
            api_key="not-needed",
            api_base_url=os.getenv(
                "LOCAL_LM_API_URL",
                "http://127.0.0.1:1234/v1"
            ),
        )

        provider.initialize()

        messages = [
            Message("user", "Hello! What's your name?"),
        ]

        response = provider.complete(messages, temperature=0.7, max_tokens=100)
        print(f"Response: {response.content}")

        provider.close()

    except Exception as e:
        print(f"Error (make sure local LM is running): {e}")


# ============================================================================
# Example 5: Using Ollama Provider Preset
# ============================================================================

def example_ollama():
    """Use Ollama provider preset"""
    print("\n" + "=" * 80)
    print("Example 5: Ollama Provider Preset")
    print("=" * 80)

    # Assumes Ollama is running at localhost:11434
    try:
        provider = OllamaProvider(
            model="llama2",  # or "neural-chat", "mistral", etc.
        )

        provider.initialize()

        messages = [
            Message("user", "Write a haiku about programming"),
        ]

        response = provider.complete(messages, temperature=0.8, max_tokens=50)
        print(f"Response:\n{response.content}")

        provider.close()

    except Exception as e:
        print(f"Error (make sure Ollama is running): {e}")


# ============================================================================
# Example 6: Streaming Response
# ============================================================================

def example_streaming():
    """Stream response from provider"""
    print("\n" + "=" * 80)
    print("Example 6: Streaming Response")
    print("=" * 80)

    try:
        provider = OpenAIProvider(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        provider.initialize()

        messages = [
            Message("user", "Write a story about an AI learning to code"),
        ]

        print("Streaming response:")
        for chunk in provider.stream(messages, max_tokens=200):
            print(chunk.content, end="", flush=True)
        print()  # New line after streaming

        provider.close()

    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# Example 7: Error Handling
# ============================================================================

def example_error_handling():
    """Demonstrate error handling"""
    print("\n" + "=" * 80)
    print("Example 7: Error Handling")
    print("=" * 80)

    from base_provider import (
        APIError,
        AuthenticationError,
        RateLimitError,
    )

    try:
        provider = OpenAIProvider(
            model="gpt-4",
            api_key="invalid-key",  # This will fail
        )

        provider.initialize()
        response = provider.complete([Message("user", "Hello")])

    except AuthenticationError as e:
        print(f"Auth error: {e.status_code} - {e}")
    except RateLimitError as e:
        print(f"Rate limit: {e}")
    except APIError as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"Other error: {e}")


# ============================================================================
# Example 8: Integration Pattern for Your Project
# ============================================================================

def example_integration_pattern():
    """
    Recommended pattern for integrating provider into your project
    """
    print("\n" + "=" * 80)
    print("Example 8: Integration Pattern for Your Project")
    print("=" * 80)

    # In your main application file:
    # 1. Load configuration from environment
    # 2. Create provider instance
    # 3. Use it throughout your app

    code_example = '''
# In your project's config.py or main.py
import os
from base_provider import OpenAIProvider, load_config_from_env

# Option 1: Direct initialization
llm_provider = OpenAIProvider(
    model=os.getenv("BOT_LLM_MODEL", "gpt-3.5-turbo"),
    api_key=os.getenv("BOT_OPENAI_API_KEY"),
    api_base_url=os.getenv(
        "BOT_OPENAI_API_BASE_URL",
        "https://api.openai.com/v1"
    ),
)

# Option 2: From environment
config = load_config_from_env(prefix="BOT")
llm_provider = OpenAIProvider(**config.to_provider_kwargs())

# Option 3: Custom function in your project
def get_llm_provider():
    """Get or create LLM provider"""
    if not hasattr(get_llm_provider, "_provider"):
        config = load_config_from_env(prefix="BOT")
        provider = OpenAIProvider(**config.to_provider_kwargs())
        provider.initialize()
        get_llm_provider._provider = provider
    return get_llm_provider._provider

# In your application code:
from base_provider import Message

provider = get_llm_provider()
messages = [
    Message("system", "You are a helpful assistant"),
    Message("user", "Hello!"),
]
response = provider.complete(messages)
print(response.content)
    '''

    print(code_example)


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "BASE PROVIDER USAGE EXAMPLES".center(78) + "║")
    print("╚" + "=" * 78 + "╝")

    # Uncomment examples you want to run

    # example_openai_official()  # Requires OPENAI_API_KEY
    # example_with_context_manager()  # Requires OPENAI_API_KEY
    # example_load_from_env()
    # example_local_lm()  # Requires local LM running
    # example_ollama()  # Requires Ollama running
    # example_streaming()  # Requires OPENAI_API_KEY
    example_error_handling()
    example_integration_pattern()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
