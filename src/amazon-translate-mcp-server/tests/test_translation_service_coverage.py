"""Additional tests to boost translation_service.py coverage.

This module contains targeted tests to improve coverage for translation_service.py,
focusing on retry logic, validation, and error scenarios.
"""

import pytest
from awslabs.amazon_translate_mcp_server.models import (
    ValidationResult,
)
from awslabs.amazon_translate_mcp_server.translation_service import TranslationService
from botocore.exceptions import ClientError
from unittest.mock import Mock, patch


class TestTranslationServiceRetryLogic:
    """Test translation service retry logic."""

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_translation_retry_on_throttling(self, mock_aws_client):
        """Test translation retry on throttling errors."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        # First call fails with throttling, second succeeds
        mock_translate_client.translate_text.side_effect = [
            ClientError(
                error_response={
                    'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}
                },
                operation_name='TranslateText',
            ),
            {
                'TranslatedText': 'Hola mundo',
                'SourceLanguageCode': 'en',
                'TargetLanguageCode': 'es',
            },
        ]
        mock_client_instance._get_client.return_value = mock_translate_client

        service = TranslationService(mock_client_instance)

        result = service.translate_text(
            text='Hello world', source_language='en', target_language='es'
        )

        assert result.translated_text == 'Hola mundo'
        assert mock_translate_client.translate_text.call_count == 2

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_language_detection_retry_logic(self, mock_aws_client):
        """Test language detection retry logic."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        # First call fails, second succeeds
        mock_translate_client.detect_dominant_language.side_effect = [
            ClientError(
                error_response={
                    'Error': {
                        'Code': 'ServiceUnavailable',
                        'Message': 'Service temporarily unavailable',
                    }
                },
                operation_name='DetectDominantLanguage',
            ),
            {'Languages': [{'LanguageCode': 'en', 'Score': 0.98}]},
        ]
        mock_client_instance._get_client.return_value = mock_translate_client

        service = TranslationService(mock_client_instance)

        result = service.detect_language('Hello world')

        assert result.detected_language == 'en'
        assert result.confidence_score == 0.98
        assert mock_translate_client.detect_dominant_language.call_count == 2


class TestTranslationValidationEdgeCases:
    """Test translation validation edge cases."""

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_validation_with_special_characters(self, mock_aws_client):
        """Test validation with special characters."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test with emojis
        result = service.validate_translation(
            source_text='Hello 😊',
            translated_text='Hola 😊',
            source_language='en',
            target_language='es',
        )

        assert isinstance(result, ValidationResult)

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_validation_with_html_content(self, mock_aws_client):
        """Test validation with HTML content."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test with HTML tags
        result = service.validate_translation(
            source_text='<p>Hello world</p>',
            translated_text='<p>Hola mundo</p>',
            source_language='en',
            target_language='es',
        )

        assert isinstance(result, ValidationResult)
        # Should detect HTML content
        assert any('html' in issue.lower() for issue in result.issues)

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_validation_with_numbers_and_dates(self, mock_aws_client):
        """Test validation with numbers and dates."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test with numbers that should be preserved
        result = service.validate_translation(
            source_text='The price is $100.50',
            translated_text='El precio es $100.50',
            source_language='en',
            target_language='es',
        )

        assert isinstance(result, ValidationResult)
        # Numbers should be preserved, so this should be valid
        assert result.is_valid

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_validation_with_urls_and_emails(self, mock_aws_client):
        """Test validation with URLs and emails."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test with URLs that should be preserved
        result = service.validate_translation(
            source_text='Visit https://example.com',
            translated_text='Visita https://example.com',
            source_language='en',
            target_language='es',
        )

        assert isinstance(result, ValidationResult)
        # URLs should be preserved


class TestTranslationServiceErrorHandling:
    """Test translation service error handling."""

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_unsupported_language_pair_error(self, mock_aws_client):
        """Test unsupported language pair error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.translate_text.side_effect = ClientError(
            error_response={
                'Error': {
                    'Code': 'UnsupportedLanguagePairException',
                    'Message': 'Language pair not supported',
                }
            },
            operation_name='TranslateText',
        )
        mock_client_instance._get_client.return_value = mock_translate_client

        service = TranslationService(mock_client_instance)

        with pytest.raises(Exception):
            service.translate_text(text='Hello', source_language='xx', target_language='yy')

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_text_size_limit_error(self, mock_aws_client):
        """Test text size limit error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.translate_text.side_effect = ClientError(
            error_response={
                'Error': {'Code': 'TextSizeLimitExceededException', 'Message': 'Text too large'}
            },
            operation_name='TranslateText',
        )
        mock_client_instance._get_client.return_value = mock_translate_client

        service = TranslationService(mock_client_instance)

        with pytest.raises(Exception):
            service.translate_text(
                text='x' * 10000,  # Very large text
                source_language='en',
                target_language='es',
            )

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_invalid_request_error(self, mock_aws_client):
        """Test invalid request error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.translate_text.side_effect = ClientError(
            error_response={
                'Error': {'Code': 'InvalidRequestException', 'Message': 'Invalid request'}
            },
            operation_name='TranslateText',
        )
        mock_client_instance._get_client.return_value = mock_translate_client

        service = TranslationService(mock_client_instance)

        with pytest.raises(Exception):
            service.translate_text(
                text='',  # Empty text
                source_language='en',
                target_language='es',
            )


class TestTranslationServiceUtilities:
    """Test translation service utility functions."""

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_confidence_score_calculation(self, mock_aws_client):
        """Test confidence score calculation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test confidence score normalization
        assert service.normalize_confidence_score(0.95) == 0.95
        assert service.normalize_confidence_score(1.5) == 1.0  # Should cap at 1.0
        assert service.normalize_confidence_score(-0.1) == 0.0  # Should floor at 0.0

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_text_preprocessing(self, mock_aws_client):
        """Test text preprocessing functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test text cleaning
        cleaned_text = service.preprocess_text('  Hello world  \n\n')
        assert cleaned_text == 'Hello world'

        # Test with special characters
        cleaned_special = service.preprocess_text('Hello\u00a0world')  # Non-breaking space
        assert cleaned_special == 'Hello world'

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_language_code_normalization(self, mock_aws_client):
        """Test language code normalization."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test case normalization
        assert service.normalize_language_code('EN') == 'en'
        assert service.normalize_language_code('Es') == 'es'
        assert service.normalize_language_code('zh-CN') == 'zh-cn'

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_translation_quality_assessment(self, mock_aws_client):
        """Test translation quality assessment."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test quality scoring
        quality_score = service.assess_translation_quality(
            source_text='Hello world',
            translated_text='Hola mundo',
            source_language='en',
            target_language='es',
        )

        assert 0.0 <= quality_score <= 1.0

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_batch_translation_preparation(self, mock_aws_client):
        """Test batch translation preparation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        service = TranslationService(mock_client_instance)

        # Test preparing texts for batch translation
        texts = ['Hello', 'World', 'How are you?']
        prepared_batch = service.prepare_batch_translation(
            texts=texts, source_language='en', target_language='es'
        )

        assert len(prepared_batch) == 3
        assert all('text' in item for item in prepared_batch)


class TestTranslationServicePerformance:
    """Test translation service performance scenarios."""

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_concurrent_translation_requests(self, mock_aws_client):
        """Test concurrent translation requests."""
        import threading
        import time

        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.translate_text.return_value = {
            'TranslatedText': 'Hola',
            'SourceLanguageCode': 'en',
            'TargetLanguageCode': 'es',
        }
        mock_client_instance._get_client.return_value = mock_translate_client

        service = TranslationService(mock_client_instance)
        results = []

        def translate_text():
            time.sleep(0.01)  # Small delay
            result = service.translate_text('Hello', 'en', 'es')
            results.append(result)

        # Create multiple threads
        threads = [threading.Thread(target=translate_text) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All translations should succeed
        assert len(results) == 5
        assert all(result.translated_text == 'Hola' for result in results)

    @patch('awslabs.amazon_translate_mcp_server.translation_service.AWSClientManager')
    def test_large_text_handling(self, mock_aws_client):
        """Test large text handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.translate_text.return_value = {
            'TranslatedText': 'Texto muy largo traducido',
            'SourceLanguageCode': 'en',
            'TargetLanguageCode': 'es',
        }
        mock_client_instance._get_client.return_value = mock_translate_client

        service = TranslationService(mock_client_instance)

        # Test with large text (but within limits)
        large_text = 'This is a very long text. ' * 100
        result = service.translate_text(large_text, 'en', 'es')

        assert result.translated_text == 'Texto muy largo traducido'
