"""Additional tests to boost language_operations.py coverage.

This module contains targeted tests to improve coverage for language_operations.py,
focusing on caching, error handling, and edge cases.
"""

import pytest
from awslabs.amazon_translate_mcp_server.language_operations import LanguageOperations
from awslabs.amazon_translate_mcp_server.models import ValidationError
from botocore.exceptions import ClientError
from unittest.mock import Mock, patch


class TestLanguageOperationsCaching:
    """Test language operations caching functionality."""

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_operations_initialization(self, mock_aws_client):
        """Test language operations initialization."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        # Test initialization
        lang_ops = LanguageOperations(mock_client_instance)
        assert lang_ops.aws_client_manager == mock_client_instance
        assert lang_ops._language_cache is None
        assert lang_ops._cache_timestamp is None

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_supported_formats(self, mock_aws_client):
        """Test supported formats functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        lang_ops = LanguageOperations(mock_client_instance)

        # Test getting supported formats
        formats = lang_ops.get_supported_formats()
        assert isinstance(formats, list)
        assert 'text/plain' in formats
        assert 'text/html' in formats

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_cache_validity_check(self, mock_aws_client):
        """Test cache validity checking."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        lang_ops = LanguageOperations(mock_client_instance)

        # Initially cache should be invalid
        assert not lang_ops._is_cache_valid()

        # Set some cache data
        from datetime import datetime
        lang_ops._language_cache = {'test': 'data'}
        lang_ops._cache_timestamp = datetime.utcnow()

        # Now cache should be valid
        assert lang_ops._is_cache_valid()


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
            operation_name='ListLanguages',
        )
        mock_client_instance.get_translate_client.return_value = mock_translate_client

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
            error_response={
                'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}
            },
            operation_name='ListLanguages',
        )
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        lang_ops = LanguageOperations(mock_client_instance)

        with pytest.raises(Exception):
            lang_ops.list_supported_languages()

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_unexpected_error_handling(self, mock_aws_client):
        """Test unexpected error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.list_languages.side_effect = Exception('Unexpected error')
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        lang_ops = LanguageOperations(mock_client_instance)

        with pytest.raises(Exception):
            lang_ops.list_supported_languages()


class TestLanguageValidationEdgeCases:
    """Test language validation edge cases."""

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_name_lookup(self, mock_aws_client):
        """Test language name lookup functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        lang_ops = LanguageOperations(mock_client_instance)

        # Test with cached data
        lang_ops._language_cache = {
            'languages': [
                {'LanguageCode': 'en', 'LanguageName': 'English'},
                {'LanguageCode': 'es', 'LanguageName': 'Spanish'},
            ]
        }
        from datetime import datetime
        lang_ops._cache_timestamp = datetime.utcnow()

        # Test successful lookup
        assert lang_ops.get_language_name('en') == 'English'
        assert lang_ops.get_language_name('es') == 'Spanish'
        
        # Test non-existent language
        assert lang_ops.get_language_name('xyz') is None

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_pair_validation_edge_cases(self, mock_aws_client):
        """Test language pair validation edge cases."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        lang_ops = LanguageOperations(mock_client_instance)

        # Test with same source and target language
        with pytest.raises(ValidationError):
            lang_ops.validate_language_pair('en', 'en')

        # Test with empty strings - should raise ValidationError
        with pytest.raises(ValidationError):
            lang_ops.validate_language_pair('', 'es')
            
        with pytest.raises(ValidationError):
            lang_ops.validate_language_pair('en', '')

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_language_name_lookup_edge_cases(self, mock_aws_client):
        """Test language name lookup edge cases."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.list_languages.return_value = {
            'Languages': [
                {'LanguageCode': 'en', 'LanguageName': 'English'},
                {'LanguageCode': 'es', 'LanguageName': 'Spanish'},
            ]
        }
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        lang_ops = LanguageOperations(mock_client_instance)

        # Test with None
        assert lang_ops.get_language_name(None) is None

        # Test with empty string
        assert lang_ops.get_language_name('') is None

        # Test with unknown language code
        assert lang_ops.get_language_name('xyz') is None


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
    def test_terminology_support_check(self, mock_aws_client):
        """Test terminology support checking."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        lang_ops = LanguageOperations(mock_client_instance)

        # Test terminology support for common language pairs
        # Auto-detect doesn't support terminology
        assert not lang_ops.is_terminology_supported('auto', 'en')
        
        # Mock validate_language_pair to return True for regular pairs
        with patch.object(lang_ops, 'validate_language_pair', return_value=True):
            assert lang_ops.is_terminology_supported('en', 'es')

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_private_methods(self, mock_aws_client):
        """Test private helper methods."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        lang_ops = LanguageOperations(mock_client_instance)

        # Test language pair format validation (private method)
        assert lang_ops._is_valid_language_pair_format('en-es')
        assert not lang_ops._is_valid_language_pair_format('invalid')
        assert not lang_ops._is_valid_language_pair_format('')

        # Test time range calculation
        from datetime import datetime, timedelta
        end_time = datetime.utcnow()
        start_time = lang_ops._calculate_start_time(end_time, '1h')
        assert isinstance(start_time, datetime)
        assert start_time < end_time


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
                {'LanguageCode': 'en'},
                {'LanguageCode': 'es'},
                {'LanguageCode': 'fr'},
                {'LanguageCode': 'de'},
            ]
        }
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        lang_ops = LanguageOperations(mock_client_instance)

        # Test getting language pairs multiple times
        pairs1 = lang_ops.list_language_pairs()
        pairs2 = lang_ops.list_language_pairs()
        
        # Should return consistent results
        assert len(pairs1) == len(pairs2)
        assert isinstance(pairs1, list)
        assert isinstance(pairs2, list)

    @patch('awslabs.amazon_translate_mcp_server.language_operations.AWSClientManager')
    def test_concurrent_cache_access(self, mock_aws_client):
        """Test concurrent cache access scenarios."""
        import threading
        import time

        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.list_languages.return_value = {'Languages': [{'LanguageCode': 'en'}]}
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        lang_ops = LanguageOperations(mock_client_instance)
        results = []

        def get_languages():
            try:
                time.sleep(0.01)  # Small delay to increase chance of race condition
                results.append(lang_ops.list_language_pairs())
            except Exception as e:
                results.append(f"Error: {e}")

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
