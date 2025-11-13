"""Additional tests to boost terminology_manager.py coverage.

This module contains targeted tests to improve coverage for terminology_manager.py,
focusing on file handling, validation, and error scenarios.
"""

import os
import pytest
import tempfile
from awslabs.amazon_translate_mcp_server.exceptions import (
    TerminologyError,
)
from awslabs.amazon_translate_mcp_server.models import (
    ValidationError,
)
from awslabs.amazon_translate_mcp_server.terminology_manager import TerminologyManager
from botocore.exceptions import ClientError
from unittest.mock import Mock, patch


class TestTerminologyFileHandling:
    """Test terminology file handling functionality."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_manager_initialization(self, mock_aws_client):
        """Test terminology manager initialization."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)
        
        # Test that the manager is properly initialized
        assert terminology_manager._aws_client_manager == mock_client_instance
        assert terminology_manager._translate_client is None

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_get_translate_client(self, mock_aws_client):
        """Test getting translate client."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance
        
        mock_translate_client = Mock()
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)
        
        # First call should create the client
        client = terminology_manager._get_translate_client()
        assert client == mock_translate_client
        
        # Second call should return the cached client
        client2 = terminology_manager._get_translate_client()
        assert client2 == mock_translate_client
        assert terminology_manager._translate_client == mock_translate_client

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_name_validation(self, mock_aws_client):
        """Test terminology name validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test valid name
        terminology_manager._validate_terminology_name('valid-name')
        
        # Test invalid names - each in separate context
        with pytest.raises(ValidationError, match="Terminology name cannot be empty"):
            terminology_manager._validate_terminology_name('')
            
        with pytest.raises(ValidationError, match="Terminology name cannot be empty"):
            terminology_manager._validate_terminology_name('   ')
            
        with pytest.raises(ValidationError, match="Terminology name cannot exceed"):
            terminology_manager._validate_terminology_name('a' * 257)  # Too long

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_language_code_validation(self, mock_aws_client):
        """Test language code validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test language code validation
        terminology_manager._validate_language_code('en', 'source_language')
        terminology_manager._validate_language_code('es', 'target_language')
        
        # Test invalid language codes - each in separate context
        with pytest.raises(ValidationError, match="source_language cannot be empty"):
            terminology_manager._validate_language_code('', 'source_language')
            
        with pytest.raises(ValidationError, match="Invalid language code format"):
            terminology_manager._validate_language_code('invalid', 'source_language')


class TestTerminologyValidationEdgeCases:
    """Test terminology validation edge cases."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_description_validation(self, mock_aws_client):
        """Test terminology description validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test valid description
        terminology_manager._validate_terminology_description('Valid description')

        # Test invalid descriptions - each in separate context
        with pytest.raises(ValidationError, match="Description must be a string"):
            terminology_manager._validate_terminology_description(123)  # Not a string

        with pytest.raises(ValidationError, match="Description cannot exceed"):
            terminology_manager._validate_terminology_description('a' * 300)  # Too long

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_data_validation(self, mock_aws_client):
        """Test terminology data validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test with invalid data type
        with pytest.raises(ValidationError, match="terminology_data must be a TerminologyData"):
            terminology_manager._validate_terminology_data("not_terminology_data")

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_constants(self, mock_aws_client):
        """Test terminology manager constants."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test constants are properly defined
        assert 'CSV' in terminology_manager.SUPPORTED_FORMATS
        assert 'TMX' in terminology_manager.SUPPORTED_FORMATS
        assert terminology_manager.MAX_TERMINOLOGY_SIZE == 10 * 1024 * 1024
        assert terminology_manager.MAX_TERMINOLOGIES == 100
        assert terminology_manager.MAX_TERM_PAIRS == 10000


class TestTerminologyErrorHandling:
    """Test terminology error handling scenarios."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_aws_client_error_handling(self, mock_aws_client):
        """Test AWS client error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.list_terminologies.side_effect = ClientError(
            error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
            operation_name='ListTerminologies',
        )
        terminology_manager = TerminologyManager(mock_client_instance)
        terminology_manager._translate_client = mock_translate_client

        with pytest.raises(Exception):
            terminology_manager.list_terminologies()

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_not_found_error(self, mock_aws_client):
        """Test terminology not found error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.get_terminology.side_effect = ClientError(
            error_response={
                'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Terminology not found'}
            },
            operation_name='GetTerminology',
        )
        terminology_manager = TerminologyManager(mock_client_instance)
        terminology_manager._translate_client = mock_translate_client

        with pytest.raises(Exception):
            terminology_manager.get_terminology('nonexistent-terminology')

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_limit_exceeded_error(self, mock_aws_client):
        """Test terminology limit exceeded error handling."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.import_terminology.side_effect = ClientError(
            error_response={
                'Error': {
                    'Code': 'LimitExceededException',
                    'Message': 'Terminology limit exceeded',
                }
            },
            operation_name='ImportTerminology',
        )
        terminology_manager = TerminologyManager(mock_client_instance)
        terminology_manager._translate_client = mock_translate_client

        # Use secure temporary file instead of hardcoded /tmp/
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with pytest.raises(Exception):
                terminology_manager.import_terminology(
                    name='test-terminology',
                    file_path=temp_path,
                    source_language='en',
                    target_languages=['es'],
                )
        finally:
            # Clean up the temporary file
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTerminologyOperationsEdgeCases:
    """Test terminology operations edge cases."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_creation_with_minimal_data(self, mock_aws_client):
        """Test terminology creation with minimal data."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.import_terminology.return_value = {
            'TerminologyProperties': {
                'Name': 'test-terminology',
                'Arn': 'arn:aws:translate:us-east-1:123456789012:terminology/test-terminology',
                'SourceLanguageCode': 'en',
                'TargetLanguageCodes': ['es'],
            }
        }
        terminology_manager = TerminologyManager(mock_client_instance)
        terminology_manager._translate_client = mock_translate_client

        # Create minimal CSV content
        csv_content = 'en,es\nhello,hola'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name

        try:
            result = terminology_manager.import_terminology(
                name='test-terminology',
                file_path=temp_file,
                source_language='en',
                target_languages=['es'],
            )

            # The method returns the ARN as a string
            assert isinstance(result, str)
            assert 'test-terminology' in result
        finally:
            os.unlink(temp_file)

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_update_scenarios(self, mock_aws_client):
        """Test terminology update scenarios."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.import_terminology.return_value = {
            'TerminologyProperties': {
                'Name': 'existing-terminology',
                'Arn': 'arn:aws:translate:us-east-1:123456789012:terminology/existing-terminology',
                'SourceLanguageCode': 'en',
                'TargetLanguageCodes': ['es', 'fr'],
            }
        }
        terminology_manager = TerminologyManager(mock_client_instance)
        terminology_manager._translate_client = mock_translate_client

        # Test updating existing terminology
        csv_content = 'en,es,fr\nhello,hola,bonjour\nworld,mundo,monde'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name

        try:
            result = terminology_manager.import_terminology(
                name='existing-terminology',
                file_path=temp_file,
                source_language='en',
                target_languages=['es', 'fr'],
            )

            # The method returns the ARN as a string
            assert isinstance(result, str)
            assert 'existing-terminology' in result
        finally:
            os.unlink(temp_file)

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_deletion_scenarios(self, mock_aws_client):
        """Test terminology deletion scenarios."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.delete_terminology.return_value = {}
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test successful deletion
        result = terminology_manager.delete_terminology('test-terminology')
        assert result is True

        # Verify delete was called
        mock_translate_client.delete_terminology.assert_called_with(Name='test-terminology')


class TestTerminologyUtilities:
    """Test terminology utility functions."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_conflict_validation(self, mock_aws_client):
        """Test terminology conflict validation functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        # Mock the get_terminology method to avoid AWS calls
        mock_translate_client = Mock()
        mock_translate_client.get_terminology.return_value = {
            'TerminologyProperties': {
                'Name': 'term1',
                'SourceLanguageCode': 'en',
                'TargetLanguageCodes': ['es'],
                'TermCount': 10,
            }
        }
        mock_client_instance.get_translate_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test conflict validation with empty list should raise ValidationError
        with pytest.raises(ValidationError, match="terminology_names must be a non-empty list"):
            terminology_manager.validate_terminology_conflicts([], 'en', 'es')

        # Test with terminology names
        conflicts = terminology_manager.validate_terminology_conflicts(['term1', 'term2'], 'en', 'es')
        assert isinstance(conflicts, dict)

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_private_methods(self, mock_aws_client):
        """Test terminology private validation methods."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test private file format detection method
        from pathlib import Path
        
        # Test CSV detection
        csv_format = terminology_manager._detect_file_format(Path('test.csv'), b'en,es\nhello,hola')
        assert csv_format == 'CSV'
        
        # Test TMX detection
        tmx_content = b'<?xml version="1.0"?><tmx><body></body></tmx>'
        tmx_format = terminology_manager._detect_file_format(Path('test.tmx'), tmx_content)
        assert tmx_format == 'TMX'
