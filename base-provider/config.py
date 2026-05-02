"""
Configuration management for LLM providers
Handles loading and validating configuration from environment variables
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ProviderType(str, Enum):
    """Supported provider types"""
    OPENAI = "openai"
    LOCAL = "local"
    ANTHROPIC = "anthropic"  # Future support
    HUGGINGFACE = "huggingface"  # Future support


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider"""

    provider_type: ProviderType
    model: str
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    organization: Optional[str] = None  # For OpenAI organization
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra: Dict[str, Any] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if isinstance(self.provider_type, str):
            self.provider_type = ProviderType(self.provider_type)

        if self.extra is None:
            self.extra = {}

        self.validate()

    def validate(self) -> None:
        """Validate the configuration"""
        if not self.model:
            raise ValueError("Model name is required")

        if self.provider_type == ProviderType.OPENAI:
            if not self.api_key:
                raise ValueError("API key is required for OpenAI provider")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_provider_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for provider initialization"""
        return {
            "model": self.model,
            "api_key": self.api_key,
            "api_base_url": self.api_base_url,
            "organization": self.organization,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            **self.extra
        }


def load_config_from_env(
    provider_type: Optional[str] = None,
    prefix: str = ""
) -> ProviderConfig:
    """
    Load provider configuration from environment variables

    Environment variable naming convention:
    - {prefix}PROVIDER_TYPE: Type of provider (openai, local, etc.)
    - {prefix}MODEL: Model name
    - {prefix}API_KEY: API key
    - {prefix}API_BASE_URL: API base URL
    - {prefix}ORGANIZATION: Organization (OpenAI)
    - {prefix}TIMEOUT: Request timeout in seconds
    - {prefix}MAX_RETRIES: Maximum retries
    - {prefix}TEMPERATURE: Default temperature
    - {prefix}MAX_TOKENS: Default max tokens

    Example with no prefix:
        PROVIDER_TYPE=openai
        MODEL=gpt-4
        API_KEY=sk-...
        API_BASE_URL=https://api.openai.com/v1

    Example with prefix "BOT_":
        BOT_PROVIDER_TYPE=openai
        BOT_MODEL=gpt-4
        BOT_API_KEY=sk-...
        BOT_API_BASE_URL=https://api.openai.com/v1

    Args:
        provider_type: Override provider type from env
        prefix: Prefix for environment variable names

    Returns:
        ProviderConfig object
    """
    prefix_str = f"{prefix}_" if prefix else ""

    # Determine provider type
    if provider_type:
        ptype = provider_type
    else:
        ptype = os.getenv(f"{prefix_str}PROVIDER_TYPE", "openai")

    # Load configuration
    config = ProviderConfig(
        provider_type=ptype,
        model=os.getenv(
            f"{prefix_str}MODEL",
            "gpt-3.5-turbo"
        ),
        api_key=os.getenv(f"{prefix_str}API_KEY"),
        api_base_url=os.getenv(
            f"{prefix_str}API_BASE_URL",
            "https://api.openai.com/v1"  # Default OpenAI URL
        ),
        organization=os.getenv(f"{prefix_str}ORGANIZATION"),
        timeout=int(os.getenv(f"{prefix_str}TIMEOUT", "30")),
        max_retries=int(os.getenv(f"{prefix_str}MAX_RETRIES", "3")),
        temperature=float(os.getenv(f"{prefix_str}TEMPERATURE", "0.7")),
        max_tokens=_parse_optional_int(
            os.getenv(f"{prefix_str}MAX_TOKENS")
        ),
    )

    return config


def load_config_from_dict(config_dict: Dict[str, Any]) -> ProviderConfig:
    """
    Load provider configuration from a dictionary

    Args:
        config_dict: Configuration dictionary

    Returns:
        ProviderConfig object
    """
    # Extract known fields
    known_fields = {
        'provider_type', 'model', 'api_key', 'api_base_url',
        'organization', 'timeout', 'max_retries', 'temperature',
        'max_tokens'
    }

    config_fields = {k: v for k, v in config_dict.items() if k in known_fields}
    extra_fields = {k: v for k, v in config_dict.items() if k not in known_fields}

    config_fields['extra'] = extra_fields

    return ProviderConfig(**config_fields)


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    """Parse optional integer from string"""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
