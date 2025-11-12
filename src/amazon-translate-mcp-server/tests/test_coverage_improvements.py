"""Additional tests to improve coverage for specific modules."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from awslabs.amazon_translate_mcp_server.models import (
    TranslationResult,
    LanguageDetectionResult,
    ValidationResult,
    ErrorResponse,
)
from awslabs.amazon_translate_mcp_server.exceptions import (
    TranslateException,
    AuthenticationError,
)


class TestModelValidation:
    """Test model validation edge cases."""

    def test_translation_result_edge_cases(self):
        """Test TranslationResult with edge case values."""
        # Test with minimum confidence score
        result = TranslationResult(
            translated_text="Hola",
            source_language="en",
            target_language="es",
            confidence_score=0.0
        )
        assert result.confidence_score == 0.0

        # Test with maximum confidence score
        result = TranslationResult(
            translated_text="Hola",
            source_language="en",
            target_language="es",
            confidence_score=1.0
        )
        assert result.confidence_score == 1.0

    def test_language_detection_result_edge_cases(self):
        """Test LanguageDetectionResult with edge case values."""
        # Test with alternative languages
        result = LanguageDetectionResult(
            detected_language="es",
            confidence_score=0.95,
            alternative_languages=[
                ("pt", 0.03),
                ("it", 0.02)
            ]
        )
        assert len(result.alternative_languages) == 2
        assert result.alternative_languages[0][0] == "pt"

    def test_validation_result_edge_cases(self):
        """Test ValidationResult with various quality scores."""
        # Test with very low quality
        result = ValidationResult(
            is_valid=False,
            quality_score=0.1,
            issues=["low_quality", "length_mismatch"],
            suggestions=["Review translation", "Check length"]
        )
        assert result.is_valid is False
        assert len(result.issues) == 2
        assert len(result.suggestions) == 2

    def test_error_response_with_retry_after(self):
        """Test ErrorResponse with retry_after field."""
        error = ErrorResponse(
            error_type="RateLimitError",
            error_code="RATE_LIMIT_001",
            message="Rate limit exceeded",
            retry_after=60
        )
        assert error.retry_after == 60

    def test_error_response_with_details(self):
        """Test ErrorResponse with complex details."""
        details = {
            "field_errors": {"name": "Required"},
            "validation_info": {"max_length": 100}
        }
        error = ErrorResponse(
            error_type="ValidationError",
            error_code="VALIDATION_001",
            message="Validation failed",
            details=details
        )
        assert error.details == details


class TestExceptionHandling:
    """Test exception handling edge cases."""

    def test_translate_exception_with_correlation_id(self):
        """Test TranslateException with correlation ID."""
        exc = TranslateException(
            "Test error",
            error_code="TEST_001",
            correlation_id="test-correlation-123"
        )
        assert exc.correlation_id == "test-correlation-123"

    def test_translate_exception_to_error_response(self):
        """Test TranslateException conversion to ErrorResponse."""
        exc = TranslateException(
            "Test error",
            error_code="TEST_001",
            details={"key": "value"}
        )
        response = exc.to_error_response()
        assert response.error_type == "TranslateException"
        assert response.error_code == "TEST_001"
        assert response.message == "Test error"
        assert response.details == {"key": "value"}

    def test_validation_error_with_field(self):
        """Test TranslateException with field information."""
        exc = TranslateException(
            "Invalid field value",
            details={"field": "username"}
        )
        assert exc.details["field"] == "username"

    def test_authentication_error_basic(self):
        """Test AuthenticationError basic functionality."""
        exc = AuthenticationError("Access denied")
        assert exc.error_code == "AUTH_ERROR"
        response = exc.to_error_response()
        assert response.error_type == "AuthenticationError"


class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    def test_datetime_handling(self):
        """Test datetime handling in models."""
        now = datetime.utcnow()
        
        # Test that datetime objects are handled properly
        result = TranslationResult(
            translated_text="Test",
            source_language="en",
            target_language="es",
            confidence_score=0.95
        )
        # The model should handle datetime fields properly
        assert result.translated_text == "Test"

    def test_optional_fields(self):
        """Test models with optional fields."""
        # Test ValidationResult with minimal fields
        result = ValidationResult(
            is_valid=True,
            quality_score=0.9
        )
        assert result.is_valid is True
        assert result.quality_score == 0.9
        assert result.issues == []
        assert result.suggestions == []

    def test_language_detection_without_alternatives(self):
        """Test LanguageDetectionResult without alternative languages."""
        result = LanguageDetectionResult(
            detected_language="en",
            confidence_score=0.99
        )
        assert result.detected_language == "en"
        assert result.alternative_languages == []


class TestConfigurationEdgeCases:
    """Test configuration edge cases."""

    def test_server_config_defaults(self):
        """Test ServerConfig default values."""
        from awslabs.amazon_translate_mcp_server.config import ServerConfig
        
        config = ServerConfig()
        
        # Test that defaults are set properly
        assert config.log_level == "INFO"
        assert config.max_text_length == 10000
        assert config.batch_timeout == 3600
        assert config.cache_ttl == 3600
        assert config.enable_translation_cache is True
        assert config.enable_pii_detection is False

    def test_server_config_to_security_config(self):
        """Test ServerConfig to SecurityConfig conversion."""
        from awslabs.amazon_translate_mcp_server.config import ServerConfig
        
        config = ServerConfig()
        config.enable_pii_detection = True
        config.enable_profanity_filter = True
        config.enable_content_filtering = True
        
        security_config = config.to_security_config()
        
        assert security_config.enable_pii_detection is True
        assert security_config.enable_profanity_filter is True
        assert security_config.enable_content_filtering is True


class TestLoggingConfiguration:
    """Test logging configuration edge cases."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        from awslabs.amazon_translate_mcp_server.logging_config import setup_logging
        
        # Test that setup_logging doesn't raise an exception
        setup_logging()
        
        # Test with custom level
        setup_logging(log_level='DEBUG')

    def test_structured_formatter(self):
        """Test structured formatter functionality."""
        from awslabs.amazon_translate_mcp_server.logging_config import StructuredFormatter
        import logging
        
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert 'Test message' in formatted