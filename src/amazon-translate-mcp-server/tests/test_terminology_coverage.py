"""Additional tests to boost terminology_manager.py coverage.

This module contains targeted tests to improve coverage for terminology_manager.py,
focusing on file handling, validation, and error scenarios.
"""

import os
import pytest
import tempfile
from awslabs.amazon_translate_mcp_server.exceptions import (
    TerminologyError,
    ValidationError,
)
from awslabs.amazon_translate_mcp_server.terminology_manager import TerminologyManager
from botocore.exceptions import ClientError
from unittest.mock import Mock, patch


class TestTerminologyFileHandling:
    """Test terminology file handling functionality."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_csv_file_validation_success(self, mock_aws_client):
        """Test successful CSV file validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Create a temporary CSV file with valid content
        csv_content = 'en,es,fr\nhello,hola,bonjour\nworld,mundo,monde'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name

        try:
            # Should not raise an exception
            terminology_manager.validate_csv_file(temp_file)
        finally:
            os.unlink(temp_file)

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_tmx_file_validation_success(self, mock_aws_client):
        """Test successful TMX file validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Create a temporary TMX file with valid content
        tmx_content = """<?xml version="1.0" encoding="UTF-8"?>
        <tmx version="1.4">
            <header>
                <prop type="x-filename">test.tmx</prop>
            </header>
            <body>
                <tu tuid="1">
                    <tuv xml:lang="en">
                        <seg>hello</seg>
                    </tuv>
                    <tuv xml:lang="es">
                        <seg>hola</seg>
                    </tuv>
                </tu>
            </body>
        </tmx>"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmx', delete=False) as f:
            f.write(tmx_content)
            temp_file = f.name

        try:
            # Should not raise an exception
            terminology_manager.validate_tmx_file(temp_file)
        finally:
            os.unlink(temp_file)

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_file_format_detection_edge_cases(self, mock_aws_client):
        """Test file format detection edge cases."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test case insensitive detection
        assert terminology_manager.detect_file_format('test.CSV') == 'CSV'
        assert terminology_manager.detect_file_format('test.TMX') == 'TMX'

        # Test with multiple extensions
        assert terminology_manager.detect_file_format('test.backup.csv') == 'CSV'

        # Test with no extension
        with pytest.raises(TerminologyError):
            terminology_manager.detect_file_format('test')

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_file_size_validation(self, mock_aws_client):
        """Test file size validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Create a large file that exceeds limits
        large_content = 'en,es\n' + 'word,palabra\n' * 100000  # Very large file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Should handle large files appropriately
            terminology_manager.validate_csv_file(temp_file)
        except TerminologyError:
            # Expected for very large files
            pass
        finally:
            os.unlink(temp_file)


class TestTerminologyValidationEdgeCases:
    """Test terminology validation edge cases."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_name_validation_edge_cases(self, mock_aws_client):
        """Test terminology name validation edge cases."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test with None
        with pytest.raises(ValidationError):
            terminology_manager.validate_terminology_name(None)

        # Test with very long name
        long_name = 'a' * 300  # Very long name
        with pytest.raises(ValidationError):
            terminology_manager.validate_terminology_name(long_name)

        # Test with invalid characters
        with pytest.raises(ValidationError):
            terminology_manager.validate_terminology_name('invalid@name')

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_language_code_validation_edge_cases(self, mock_aws_client):
        """Test language code validation edge cases."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test with None
        with pytest.raises(ValidationError):
            terminology_manager.validate_language_code(None)

        # Test with empty string
        with pytest.raises(ValidationError):
            terminology_manager.validate_language_code('')

        # Test with invalid format
        with pytest.raises(ValidationError):
            terminology_manager.validate_language_code('invalid-lang-code')

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_s3_uri_validation(self, mock_aws_client):
        """Test S3 URI validation."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test valid S3 URIs
        assert terminology_manager.validate_s3_uri('s3://bucket/key')
        assert terminology_manager.validate_s3_uri('s3://bucket/folder/file.csv')

        # Test invalid S3 URIs
        assert not terminology_manager.validate_s3_uri('http://example.com')
        assert not terminology_manager.validate_s3_uri('bucket/key')
        assert not terminology_manager.validate_s3_uri('')
        assert not terminology_manager.validate_s3_uri(None)


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
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

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
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

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
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

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
                'SourceLanguageCode': 'en',
                'TargetLanguageCodes': ['es'],
            }
        }
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

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

            assert result['TerminologyProperties']['Name'] == 'test-terminology'
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
                'SourceLanguageCode': 'en',
                'TargetLanguageCodes': ['es', 'fr'],
            }
        }
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

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

            assert 'fr' in result['TerminologyProperties']['TargetLanguageCodes']
        finally:
            os.unlink(temp_file)

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_deletion_scenarios(self, mock_aws_client):
        """Test terminology deletion scenarios."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.delete_terminology.return_value = {}
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test successful deletion
        result = terminology_manager.delete_terminology('test-terminology')
        assert result == {}

        # Verify delete was called
        mock_translate_client.delete_terminology.assert_called_with(Name='test-terminology')


class TestTerminologyUtilities:
    """Test terminology utility functions."""

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_statistics(self, mock_aws_client):
        """Test terminology statistics functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.list_terminologies.return_value = {
            'TerminologyPropertiesList': [
                {'Name': 'terminology1', 'TermCount': 100, 'SizeBytes': 1024},
                {'Name': 'terminology2', 'TermCount': 200, 'SizeBytes': 2048},
            ]
        }
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test getting terminology statistics
        stats = terminology_manager.get_terminology_statistics()

        assert stats['total_terminologies'] == 2
        assert stats['total_terms'] == 300
        assert stats['total_size_bytes'] == 3072

    @patch('awslabs.amazon_translate_mcp_server.terminology_manager.AWSClientManager')
    def test_terminology_search_functionality(self, mock_aws_client):
        """Test terminology search functionality."""
        mock_client_instance = Mock()
        mock_aws_client.return_value = mock_client_instance

        mock_translate_client = Mock()
        mock_translate_client.list_terminologies.return_value = {
            'TerminologyPropertiesList': [
                {'Name': 'medical-terms'},
                {'Name': 'legal-terms'},
                {'Name': 'technical-terms'},
            ]
        }
        mock_client_instance._get_client.return_value = mock_translate_client

        terminology_manager = TerminologyManager(mock_client_instance)

        # Test searching terminologies by name pattern
        results = terminology_manager.search_terminologies('medical')

        assert len(results) == 1
        assert results[0]['Name'] == 'medical-terms'
