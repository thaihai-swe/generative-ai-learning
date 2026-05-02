"""
Configuration module for AI Meeting Assistant
Manages LLM provider setup using base-provider
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add base-provider to path if it's in parent directory
base_provider_path = Path(__file__).parent.parent.parent / "base-provider"
if base_provider_path.exists():
    sys.path.insert(0, str(base_provider_path.parent))

try:
    from base_provider import load_config_from_env, OpenAIProvider
except ImportError:
    raise ImportError(
        "base-provider is required. Please ensure it's in the parent directory "
        "or install it with: pip install -e ../base-provider"
    )


def get_llm_provider():
    """
    Get or create the LLM provider instance (singleton pattern)

    Loads configuration from environment variables:
    - MEETING_ASSISTANT_PROVIDER_TYPE: Type of provider (default: openai)
    - MEETING_ASSISTANT_MODEL: Model to use (default: gpt-3.5-turbo)
    - MEETING_ASSISTANT_API_KEY: API key
    - MEETING_ASSISTANT_API_BASE_URL: API base URL (optional)
    - MEETING_ASSISTANT_TEMPERATURE: Temperature (default: 0.5)
    - MEETING_ASSISTANT_MAX_TOKENS: Max tokens (default: varies)

    Or use OPENAI_API_KEY for backward compatibility.

    Returns:
        LLMProvider: The configured provider instance

    Raises:
        ImportError: If base-provider is not available
        ConfigurationError: If configuration is invalid
    """
    if not hasattr(get_llm_provider, "_provider"):
        load_dotenv()

        # Try to load with prefix, fall back to old format for backward compatibility
        try:
            config = load_config_from_env(prefix="MEETING_ASSISTANT")
        except Exception:
            # Fall back to old environment variables
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key found. Please set either "
                    "MEETING_ASSISTANT_API_KEY or OPENAI_API_KEY in .env"
                )

            from base_provider import ProviderConfig
            config = ProviderConfig(
                provider_type="openai",
                model=os.getenv("MEETING_ASSISTANT_MODEL", "gpt-3.5-turbo"),
                api_key=api_key,
            )

        provider = OpenAIProvider(**config.to_provider_kwargs())
        provider.initialize()
        get_llm_provider._provider = provider

    return get_llm_provider._provider


def close_llm_provider():
    """Close the LLM provider and clean up resources"""
    if hasattr(get_llm_provider, "_provider"):
        provider = get_llm_provider._provider
        if provider:
            provider.close()
        delattr(get_llm_provider, "_provider")


def reset_provider():
    """Reset the provider (useful for testing)"""
    close_llm_provider()
