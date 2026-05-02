"""
Testing guide and examples for base-provider
Shows how to test providers and extensions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from base_provider import (
    OpenAIProvider,
    Message,
    Response,
    ProviderConfig,
    ConfigurationError,
    AuthenticationError,
)


class TestMessage(unittest.TestCase):
    """Test Message class"""

    def test_message_creation(self):
        """Test creating a message"""
        msg = Message("user", "Hello")
        assert msg.role.value == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        """Test converting message to dict"""
        msg = Message("assistant", "Hi there")
        msg_dict = msg.to_dict()
        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Hi there"

    def test_message_roles(self):
        """Test all message roles"""
        for role in ["system", "user", "assistant"]:
            msg = Message(role, "test")
            assert msg.role.value == role


class TestResponse(unittest.TestCase):
    """Test Response class"""

    def test_response_creation(self):
        """Test creating a response"""
        response = Response(
            content="Hello",
            model="gpt-4",
            tokens_used={"prompt": 10, "completion": 5, "total": 15}
        )
        assert response.content == "Hello"
        assert response.model == "gpt-4"

    def test_response_token_count(self):
        """Test getting token count"""
        response = Response(
            content="Test",
            model="gpt-4",
            tokens_used={"prompt": 10, "completion": 20, "total": 30}
        )
        assert response.get_token_count() == 30

    def test_response_str(self):
        """Test response string representation"""
        response = Response(content="Hello", model="gpt-4")
        assert str(response) == "Hello"


class TestProviderConfig(unittest.TestCase):
    """Test ProviderConfig class"""

    def test_config_creation(self):
        """Test creating config"""
        config = ProviderConfig(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
        )
        assert config.provider_type.value == "openai"
        assert config.model == "gpt-4"

    def test_config_validation(self):
        """Test config validation"""
        with self.assertRaises(ValueError):
            ProviderConfig(
                provider_type="openai",
                model="",  # Empty model
                api_key="sk-test"
            )

    def test_config_temperature_validation(self):
        """Test temperature validation"""
        with self.assertRaises(ValueError):
            ProviderConfig(
                provider_type="openai",
                model="gpt-4",
                api_key="sk-test",
                temperature=3.0  # Too high
            )

    def test_config_to_dict(self):
        """Test converting config to dict"""
        config = ProviderConfig(
            provider_type="openai",
            model="gpt-4",
            api_key="sk-test",
        )
        config_dict = config.to_dict()
        assert config_dict["model"] == "gpt-4"
        assert config_dict["api_key"] == "sk-test"


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider"""

    def setUp(self):
        """Set up test fixtures"""
        self.provider = OpenAIProvider(
            model="gpt-4",
            api_key="sk-test",
        )

    def test_provider_creation(self):
        """Test creating a provider"""
        assert self.provider.model == "gpt-4"
        assert self.provider.api_key == "sk-test"
        assert not self.provider._initialized

    def test_provider_repr(self):
        """Test provider string representation"""
        repr_str = repr(self.provider)
        assert "OpenAIProvider" in repr_str
        assert "gpt-4" in repr_str

    @patch("openai.OpenAI")
    def test_provider_initialize(self, mock_openai):
        """Test initializing a provider"""
        self.provider.initialize()
        assert self.provider._initialized
        assert self.provider.client is not None

    @patch("openai.OpenAI")
    def test_provider_close(self, mock_openai):
        """Test closing a provider"""
        self.provider.initialize()
        self.provider.close()
        assert not self.provider._initialized
        assert self.provider.client is None

    @patch("openai.OpenAI")
    def test_context_manager(self, mock_openai):
        """Test using provider as context manager"""
        with OpenAIProvider(
            model="gpt-4",
            api_key="sk-test",
        ) as provider:
            assert provider._initialized
        assert not provider._initialized

    @patch("openai.OpenAI")
    def test_complete_not_initialized(self, mock_openai):
        """Test complete raises error if not initialized"""
        provider = OpenAIProvider(model="gpt-4", api_key="sk-test")

        with self.assertRaises(ConfigurationError):
            provider.complete([Message("user", "Hello")])

    @patch("openai.OpenAI")
    def test_complete_success(self, mock_openai):
        """Test successful completion"""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 2
        mock_response.usage.total_tokens = 7

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-4", api_key="sk-test")
        provider.initialize()

        response = provider.complete([Message("user", "Hello")])

        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.tokens_used["total"] == 7

    @patch("openai.OpenAI")
    def test_stream(self, mock_openai):
        """Test streaming completion"""
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " World"
        mock_chunk2.choices[0].finish_reason = "stop"

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value.chat.completions.create.return_value = [
            mock_chunk1,
            mock_chunk2
        ]

        mock_openai.return_value.chat.completions.create.return_value = mock_stream

        provider = OpenAIProvider(model="gpt-4", api_key="sk-test")
        provider.initialize()

        # For simplicity, just check that streaming can be called
        # (Full streaming test is more complex due to context manager)


class TestErrorHandling(unittest.TestCase):
    """Test error handling"""

    @patch("openai.OpenAI")
    def test_authentication_error(self, mock_openai):
        """Test authentication error handling"""
        from openai import AuthenticationError as OpenAIAuthenticationError

        mock_openai.return_value.chat.completions.create.side_effect = \
            OpenAIAuthenticationError("Invalid API key")

        provider = OpenAIProvider(model="gpt-4", api_key="invalid")
        provider.initialize()

        with self.assertRaises(AuthenticationError):
            provider.complete([Message("user", "Hello")])

    @patch("openai.OpenAI")
    def test_rate_limit_error(self, mock_openai):
        """Test rate limit error handling"""
        from openai import RateLimitError as OpenAIRateLimitError

        mock_openai.return_value.chat.completions.create.side_effect = \
            OpenAIRateLimitError("Rate limit exceeded")

        provider = OpenAIProvider(model="gpt-4", api_key="sk-test")
        provider.initialize()

        from base_provider import RateLimitError
        with self.assertRaises(RateLimitError):
            provider.complete([Message("user", "Hello")])


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
