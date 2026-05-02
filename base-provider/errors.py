"""
Custom exceptions for LLM providers
"""


class ProviderError(Exception):
    """Base exception for provider errors"""
    pass


class ConfigurationError(ProviderError):
    """Raised when provider configuration is invalid"""
    pass


class APIError(ProviderError):
    """Raised when API call fails"""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        """
        Initialize API error

        Args:
            message: Error message
            status_code: HTTP status code (if applicable)
            response: Raw API response (if applicable)
        """
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails"""
    pass


class ModelNotFoundError(APIError):
    """Raised when requested model is not found"""
    pass


class StreamingError(ProviderError):
    """Raised when streaming fails"""
    pass
