"""Additional tests to boost language_operations.py coverage.

This module contains targeted tests to improve coverage for language_operations.py,
focusing on caching, error handling, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch
from awslabs.amazon_translate_mcp_server.language_operations import LanguageOperations
from awslabs.amazon_translate_mcp_server.exceptions import (
    TranslationError,
    ValidationError,
    ServiceUnavailableError
)
from botocore.exceptions import ClientError


class TestLanguageOperationsCaching:
    """Test language operations caching functionality."""

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_cache_initialization(self, mock_aws_client):
        """Test cache initialization with different TTL values."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        # Test with custom cache TTL
        lang_ops = LanguageOperations(mock_client_instance, cache_ttl=300)
        assert lang_ops._cache_ttl == 300
        
        # Test with default cache TTL
        lang_ops_default = LanguageOperations(mock_client_instance)
        assert lang_ops_default._cache_ttl == 3600  # Default 1 hour

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_cache_hit_scenarios(self, mock_aws_client):
        """Test cache hit scenarios."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.return_value = {
            'Languages': [
                {'LanguageCode': 'en', 'LanguageName': 'English'},
                {'LanguageCode': 'es', 'LanguageName': 'Spanish'}
            ]
        }
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # First call should hit AWS
        result1 = lang_ops.list_supported_languages()
        assert len(result1) == 2
        
        # Second call should hit cache
        result2 = lang_ops.list_supported_languages()
        assert len(result2) == 2
        
        # AWS should only be called once
        assert mock_translate_client.list_languages.call_count == 1

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_cache_invalidation(self, mock_aws_client):
        """Test cache invalidation functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.return_value = {
            'Languages': [{'LanguageCode': 'en'}]
        }
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # Populate cache
        lang_ops.list_supported_languages()
        
        # Invalidate cache
        lang_ops.invalidate_cache()
        
        # Next call should hit AWS again
        lang_ops.list_supported_languages()
        
        # AWS should be called twice
        assert mock_translate_client.list_languages.call_count == 2


class TestLanguageOperationsErrorHandling:
    """Test language operations error handling."""

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_aws_client_error_handling(self, mock_aws_client):
        """Test AWS client error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.side_effect = ClientError(
            error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
            operation_name='ListLanguages'
        )
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        with pytest.raises(Exception):
            lang_ops.list_supported_languages()

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_service_unavailable_error(self, mock_aws_client):
        """Test service unavailable error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.side_effect = ClientError(
            error_response={'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
            operation_name='ListLanguages'
        )
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        with pytest.raises(Exception):
            lang_ops.list_supported_languages()

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_unexpected_error_handling(self, mock_aws_client):
        """Test unexpected error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.side_effect = Exception("Unexpected error")
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        with pytest.raises(Exception):
            lang_ops.list_supported_languages()


class TestLanguageValidationEdgeCases:
    """Test language validation edge cases."""

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_code_normalization(self, mock_aws_client):
        """Test language code normalization."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # Test case insensitive validation
        assert lang_ops.is_valid_language_code('EN') == lang_ops.is_valid_language_code('en')
        assert lang_ops.is_valid_language_code('Es') == lang_ops.is_valid_language_code('es')

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_pair_validation_edge_cases(self, mock_aws_client):
        """Test language pair validation edge cases."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # Test with None values
        assert not lang_ops.validate_language_pair(None, 'es')
        assert not lang_ops.validate_language_pair('en', None)
        assert not lang_ops.validate_language_pair(None, None)
        
        # Test with empty strings
        assert not lang_ops.validate_language_pair('', 'es')
        assert not lang_ops.validate_language_pair('en', '')
        assert not lang_ops.validate_language_pair('', '')

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_name_lookup_edge_cases(self, mock_aws_client):
        """Test language name lookup edge cases."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.return_value = {
            'Languages': [
                {'LanguageCode': 'en', 'LanguageName': 'English'},
                {'LanguageCode': 'es', 'LanguageName': 'Spanish'}
            ]
        }
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # Test with None
        assert lang_ops.get_language_name(None) is None
        
        # Test with empty string
        assert lang_ops.get_language_name('') == ''
        
        # Test with unknown language code
        assert lang_ops.get_language_name('xyz') == 'xyz'


class TestLanguageOperationsUtilities:
    """Test language operations utility functions."""

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_supported_formats_functionality(self, mock_aws_client):
        """Test supported formats functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        formats = lang_ops.get_supported_formats()
        assert isinstance(formats, list)
        assert 'text/plain' in formats
        assert 'text/html' in formats

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_pair_format_validation(self, mock_aws_client):
        """Test language pair format validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # Test valid formats
        assert lang_ops.is_valid_language_pair_format('en-es')
        assert lang_ops.is_valid_language_pair_format('en_US-es_ES')
        
        # Test invalid formats
        assert not lang_ops.is_valid_language_pair_format('invalid')
        assert not lang_ops.is_valid_language_pair_format('en')
        assert not lang_ops.is_valid_language_pair_format('')
        assert not lang_ops.is_valid_language_pair_format(None)

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_detection_confidence(self, mock_aws_client):
        """Test language detection confidence handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # Test confidence threshold validation
        assert lang_ops.is_confidence_acceptable(0.95, 0.8)
        assert not lang_ops.is_confidence_acceptable(0.7, 0.8)
        assert not lang_ops.is_confidence_acceptable(None, 0.8)


class TestLanguageOperationsPerformance:
    """Test language operations performance scenarios."""

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_bulk_language_validation(self, mock_aws_client):
        """Test bulk language validation performance."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.return_value = {
            'Languages': [
                {'LanguageCode': 'en'}, {'LanguageCode': 'es'}, 
                {'LanguageCode': 'fr'}, {'LanguageCode': 'de'}
            ]
        }
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        
        # Test validating multiple language codes
        language_codes = ['en', 'es', 'fr', 'de', 'invalid']
        valid_codes = [code for code in language_codes if lang_ops.is_valid_language_code(code)]
        
        assert len(valid_codes) == 4
        assert 'invalid' not in valid_codes

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_concurrent_cache_access(self, mock_aws_client):
        """Test concurrent cache access scenarios."""
        import threading
        import time
        
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_translate_client.list_languages.return_value = {
            'Languages': [{'LanguageCode': 'en'}]
        }
        mock_client_instance._get_client.return_value = mock_translate_client
        
        lang_ops = LanguageOperations(mock_client_instance)
        results = []
        
        def get_languages():
            time.sleep(0.01)  # Small delay to increase chance of race condition
            results.append(lang_ops.list_supported_languages())
        
        # Create multiple threads accessing cache concurrently
        threads = [threading.Thread(target=get_languages) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should get the same result
        assert len(results) == 5
        assert all(len(result) == 1 for result in results)
        
        # AWS should only be called once due to caching
        assert mock_translate_client.list_languages.call_count == 1