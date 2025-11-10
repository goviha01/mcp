"""Unit tests for secure translation service integration.

This module tests the integration of security features with translation services,
demonstrating how security features work in practice.
"""

import pytest
from awslabs.amazon_translate_mcp_server.config import ServerConfig, reset_config, set_config
from awslabs.amazon_translate_mcp_server.exceptions import SecurityError, ValidationError
from awslabs.amazon_translate_mcp_server.secure_translation_service import (
    SecureTranslationService,
    SecurityIntegrationExample,
)
from unittest.mock import patch


class TestSecureTranslationService:
    """Test secure translation service functionality."""

    @pytest.fixture
    def secure_service(self):
        """Create secure translation service with test config."""
        config = ServerConfig(
            enable_pii_detection=True,
            enable_profanity_filter=True,
            enable_audit_logging=True,
            max_text_length=1000,
        )
        set_config(config)

        with patch(
            'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
        ):
            service = SecureTranslationService()
            yield service

        reset_config()

    def test_translate_text_secure_basic(self, secure_service):
        """Test basic secure translation."""
        result, pii_findings = secure_service.translate_text_secure(
            text='Hello world', source_language='en', target_language='es'
        )

        assert result.translated_text == 'Texto traducido simulado'
        assert result.source_language == 'en'
        assert result.target_language == 'es'
        assert len(pii_findings) == 0

    def test_translate_text_secure_with_pii(self, secure_service):
        """Test secure translation with PII detection."""
        result, pii_findings = secure_service.translate_text_secure(
            text='Contact me at john@example.com', source_language='en', target_language='es'
        )

        assert result.translated_text == 'Texto traducido simulado'
        assert len(pii_findings) == 1
        assert pii_findings[0]['type'] == 'EMAIL'
        assert 'PII_MASKED' in result.applied_terminologies

    def test_translate_text_secure_with_terminology(self, secure_service):
        """Test secure translation with terminology."""
        result, pii_findings = secure_service.translate_text_secure(
            text='Hello world',
            source_language='en',
            target_language='es',
            terminology_names=['medical', 'technical'],
        )

        assert 'medical' in result.applied_terminologies
        assert 'technical' in result.applied_terminologies

    def test_translate_text_secure_validation_error(self, secure_service):
        """Test secure translation with validation error."""
        with pytest.raises(ValidationError, match='cannot be empty'):
            secure_service.translate_text_secure(
                text='', source_language='en', target_language='es'
            )

    def test_translate_text_secure_invalid_language(self, secure_service):
        """Test secure translation with invalid language code."""
        with pytest.raises(ValidationError, match='Invalid.*format'):
            secure_service.translate_text_secure(
                text='Hello world', source_language='invalid', target_language='es'
            )

    def test_translate_text_secure_blocked_content(self):
        """Test secure translation with blocked content."""
        config = ServerConfig(enable_content_filtering=True, blocked_patterns=[r'confidential'])
        set_config(config)

        try:
            with patch(
                'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
            ):
                service = SecureTranslationService()

                with pytest.raises(SecurityError, match='blocked content pattern'):
                    service.translate_text_secure(
                        text='This is confidential information',
                        source_language='en',
                        target_language='es',
                    )
        finally:
            reset_config()

    def test_detect_language_secure(self, secure_service):
        """Test secure language detection."""
        result, pii_findings = secure_service.detect_language_secure(
            text='Hello world, this is English text'
        )

        assert result.detected_language == 'en'
        assert result.confidence_score == 0.9
        assert len(pii_findings) == 0

    def test_detect_language_secure_with_pii(self, secure_service):
        """Test secure language detection with PII."""
        result, pii_findings = secure_service.detect_language_secure(
            text='Hello, call me at 555-123-4567'
        )

        assert result.detected_language == 'en'
        assert len(pii_findings) == 1
        assert pii_findings[0]['type'] == 'PHONE'

    def test_validate_translation_secure(self, secure_service):
        """Test secure translation validation."""
        result = secure_service.validate_translation_secure(
            original_text='Hello world',
            translated_text='Hola mundo',
            source_language='en',
            target_language='es',
        )

        assert result.is_valid is True
        assert result.quality_score == 0.9
        assert len(result.issues) == 0

    def test_validate_translation_secure_poor_quality(self, secure_service):
        """Test secure translation validation with poor quality."""
        result = secure_service.validate_translation_secure(
            original_text='Hello world',
            translated_text='X',  # Very short translation
            source_language='en',
            target_language='es',
        )

        assert result.is_valid is False
        assert result.quality_score == 0.4
        assert len(result.issues) > 0

    @patch('awslabs.amazon_translate_mcp_server.secure_translation_service.time.time')
    def test_translation_timing_logging(self, mock_time, secure_service):
        """Test that translation timing is logged correctly."""
        # Provide enough values for all time.time() calls (including logging system calls)
        mock_time.side_effect = [1000.0, 1001.5, 1002.0, 1003.0, 1004.0, 1005.0]

        with patch.object(secure_service.security_manager, 'log_operation') as mock_log:
            result, _ = secure_service.translate_text_secure(
                text='Hello world', source_language='en', target_language='es'
            )

            # Should log start and complete operations
            assert mock_log.call_count == 2

            # Check start operation log
            start_call = mock_log.call_args_list[0]
            assert start_call[1]['operation'] == 'translate_text_start'
            assert start_call[1]['success'] is True

            # Check complete operation log
            complete_call = mock_log.call_args_list[1]
            assert complete_call[1]['operation'] == 'translate_text_complete'
            assert complete_call[1]['success'] is True

    def test_error_logging(self, secure_service):
        """Test that errors are properly logged."""
        with patch.object(secure_service.security_manager, 'log_operation') as mock_log:
            with pytest.raises(ValidationError):
                secure_service.translate_text_secure(
                    text='',  # Empty text will cause validation error
                    source_language='en',
                    target_language='es',
                )

            # Should log error operation
            mock_log.assert_called_once()
            call_args = mock_log.call_args[1]
            assert call_args['operation'] == 'translate_text_error'
            assert call_args['success'] is False
            # The error_code comes from the exception's error_code attribute
            assert 'error_code' in call_args


class TestSecurityIntegrationExample:
    """Test security integration examples."""

    @pytest.fixture
    def integration_example(self):
        """Create integration example with test config."""
        config = ServerConfig(
            enable_pii_detection=True, enable_profanity_filter=True, enable_audit_logging=True
        )
        set_config(config)

        with patch(
            'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
        ):
            example = SecurityIntegrationExample()
            yield example

        reset_config()

    def test_secure_translation_workflow(self, integration_example, capsys):
        """Test complete secure translation workflow."""
        integration_example.example_secure_translation_workflow()

        captured = capsys.readouterr()
        assert 'Translation:' in captured.out
        assert 'PII findings:' in captured.out
        assert 'Detected language:' in captured.out
        assert 'Translation valid:' in captured.out

    def test_batch_security_integration(self, integration_example, capsys):
        """Test batch security integration."""
        secure_texts, pii_findings = integration_example.example_batch_security_integration()

        # Should process some texts successfully
        assert len(secure_texts) > 0

        # Should find PII in the test data
        assert len(pii_findings) > 0

        captured = capsys.readouterr()
        assert 'Processed' in captured.out
        assert 'PII findings:' in captured.out

    def test_batch_security_with_blocked_content(self):
        """Test batch processing with blocked content."""
        config = ServerConfig(
            enable_pii_detection=True,
            enable_content_filtering=True,
            blocked_patterns=[r'confidential'],
        )
        set_config(config)

        try:
            with patch(
                'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
            ):
                example = SecurityIntegrationExample()

                secure_texts, pii_findings = example.example_batch_security_integration()

                # Should process fewer texts due to blocked content
                assert len(secure_texts) < 4  # Original has 4 texts, one should be blocked

        finally:
            reset_config()


class TestSecurityFeatureIntegration:
    """Test integration of individual security features."""

    def test_pii_detection_integration(self):
        """Test PII detection integration in translation workflow."""
        config = ServerConfig(enable_pii_detection=True)
        set_config(config)

        try:
            with patch(
                'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
            ):
                service = SecureTranslationService()

                # Test with email
                result, pii_findings = service.translate_text_secure(
                    text='Contact john.doe@example.com', source_language='en', target_language='es'
                )

                assert len(pii_findings) == 1
                assert pii_findings[0]['type'] == 'EMAIL'
                assert 'PII_MASKED' in result.applied_terminologies

        finally:
            reset_config()

    def test_profanity_filter_integration(self):
        """Test profanity filter integration in translation workflow."""
        config = ServerConfig(enable_profanity_filter=True)
        set_config(config)

        try:
            with patch(
                'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
            ):
                service = SecureTranslationService()

                # Test with profanity (using mild example)
                result, pii_findings = service.translate_text_secure(
                    text='This is damn good', source_language='en', target_language='es'
                )

                # Should complete successfully with filtered content
                assert result.translated_text == 'Texto traducido simulado'

        finally:
            reset_config()

    def test_audit_logging_integration(self):
        """Test audit logging integration."""
        config = ServerConfig(enable_audit_logging=True)
        set_config(config)

        try:
            with patch(
                'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
            ):
                service = SecureTranslationService()

                with patch.object(
                    service.security_manager.audit_logger, 'log_translation'
                ) as mock_log:
                    service.translate_text_secure(
                        text='Hello world',
                        source_language='en',
                        target_language='es',
                        user_id='test_user',
                    )

                    # Should log start and complete operations
                    assert mock_log.call_count == 2

                    # Check that user_id is logged
                    for call in mock_log.call_args_list:
                        assert call[1]['user_id'] == 'test_user'

        finally:
            reset_config()

    def test_comprehensive_security_integration(self):
        """Test all security features working together."""
        config = ServerConfig(
            enable_pii_detection=True,
            enable_profanity_filter=True,
            enable_audit_logging=True,
            max_text_length=100,
        )
        set_config(config)

        try:
            with patch(
                'awslabs.amazon_translate_mcp_server.secure_translation_service.AWSClientManager'
            ):
                service = SecureTranslationService()

                with patch.object(
                    service.security_manager.audit_logger, 'log_translation'
                ) as mock_log:
                    # Test with text containing PII and profanity
                    result, pii_findings = service.translate_text_secure(
                        text='This damn email john@example.com is good',
                        source_language='en',
                        target_language='es',
                        user_id='test_user',
                    )

                    # Should complete successfully
                    assert result.translated_text == 'Texto traducido simulado'

                    # Should detect PII
                    assert len(pii_findings) == 1
                    assert pii_findings[0]['type'] == 'EMAIL'
                    assert 'PII_MASKED' in result.applied_terminologies

                    # Should log operations
                    assert mock_log.call_count == 2

        finally:
            reset_config()
