# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for TerminologyManager.

This module contains comprehensive unit tests for the TerminologyManager class,
covering all terminology operations including listing, creating, importing,
retrieving, and validation functionality.
"""

import csv
import io
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from botocore.exceptions import ClientError, BotoCoreError

from awslabs.amazon_translate_mcp_server.terminology_manager import TerminologyManager
from awslabs.amazon_translate_mcp_server.models import (
    TerminologyData,
    TerminologyDetails,
    TerminologySummary,
    TerminologyError,
    ValidationError,
    AuthenticationError,
    ServiceUnavailableError,
    QuotaExceededError,
    RateLimitError
)


class TestTerminologyManager:
    """Test cases for TerminologyManager."""
    
    @pytest.fixture
    def mock_aws_client_manager(self):
        """Create a mock AWS client manager."""
        manager = Mock()
        manager.get_translate_client.return_value = Mock()
        return manager
    
    @pytest.fixture
    def terminology_manager(self, mock_aws_client_manager):
        """Create a TerminologyManager instance with mocked dependencies."""
        return TerminologyManager(mock_aws_client_manager)
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV terminology content."""
        csv_data = "en,es\nhello,hola\nworld,mundo\ngoodbye,adiós"
        return csv_data.encode('utf-8')
    
    @pytest.fixture
    def sample_tmx_content(self):
        """Sample TMX terminology content."""
        tmx_data = '''<?xml version="1.0" encoding="UTF-8"?>
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
    <tu tuid="2">
      <tuv xml:lang="en">
        <seg>world</seg>
      </tuv>
      <tuv xml:lang="es">
        <seg>mundo</seg>
      </tuv>
    </tu>
  </body>
</tmx>'''
        return tmx_data.encode('utf-8')
    
    def test_init(self, mock_aws_client_manager):
        """Test TerminologyManager initialization."""
        manager = TerminologyManager(mock_aws_client_manager)
        assert manager._aws_client_manager == mock_aws_client_manager
        assert manager._translate_client is None
    
    def test_get_translate_client(self, terminology_manager, mock_aws_client_manager):
        """Test getting translate client."""
        mock_client = Mock()
        mock_aws_client_manager.get_translate_client.return_value = mock_client
        
        client = terminology_manager._get_translate_client()
        assert client == mock_client
        assert terminology_manager._translate_client == mock_client
        
        # Second call should return cached client
        client2 = terminology_manager._get_translate_client()
        assert client2 == mock_client
        mock_aws_client_manager.get_translate_client.assert_called_once()
    
    def test_list_terminologies_success(self, terminology_manager):
        """Test successful terminology listing."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        # Mock AWS response
        mock_response = {
            'TerminologyPropertiesList': [
                {
                    'Name': 'medical-terms',
                    'Description': 'Medical terminology',
                    'SourceLanguageCode': 'en',
                    'TargetLanguageCodes': ['es', 'fr'],
                    'TermCount': 100,
                    'CreatedAt': datetime(2023, 1, 1)
                },
                {
                    'Name': 'legal-terms',
                    'Description': 'Legal terminology',
                    'SourceLanguageCode': 'en',
                    'TargetLanguageCodes': ['de'],
                    'TermCount': 50,
                    'CreatedAt': datetime(2023, 2, 1)
                }
            ],
            'NextToken': 'next-token-123'
        }
        mock_client.list_terminologies.return_value = mock_response
        
        result = terminology_manager.list_terminologies(max_results=10)
        
        assert len(result['terminologies']) == 2
        assert result['next_token'] == 'next-token-123'
        
        # Check first terminology
        term1 = result['terminologies'][0]
        assert isinstance(term1, TerminologySummary)
        assert term1.name == 'medical-terms'
        assert term1.description == 'Medical terminology'
        assert term1.source_language == 'en'
        assert term1.target_languages == ['es', 'fr']
        assert term1.term_count == 100
        
        mock_client.list_terminologies.assert_called_once_with(MaxResults=10)
    
    def test_list_terminologies_with_next_token(self, terminology_manager):
        """Test terminology listing with pagination token."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_response = {
            'TerminologyPropertiesList': [],
            'NextToken': None
        }
        mock_client.list_terminologies.return_value = mock_response
        
        result = terminology_manager.list_terminologies(max_results=25, next_token='token-123')
        
        assert result['terminologies'] == []
        assert result['next_token'] is None
        
        mock_client.list_terminologies.assert_called_once_with(
            MaxResults=25,
            NextToken='token-123'
        )
    
    def test_list_terminologies_validation_errors(self, terminology_manager):
        """Test validation errors in list_terminologies."""
        # Invalid max_results
        with pytest.raises(ValidationError) as exc_info:
            terminology_manager.list_terminologies(max_results=0)
        assert "max_results must be an integer between 1 and 500" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            terminology_manager.list_terminologies(max_results=501)
        assert "max_results must be an integer between 1 and 500" in str(exc_info.value)
        
        # Invalid next_token
        with pytest.raises(ValidationError) as exc_info:
            terminology_manager.list_terminologies(next_token=123)
        assert "next_token must be a string" in str(exc_info.value)
    
    def test_list_terminologies_client_errors(self, terminology_manager):
        """Test client errors in list_terminologies."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        # Access denied error
        mock_client.list_terminologies.side_effect = ClientError(
            {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}},
            'ListTerminologies'
        )
        
        with pytest.raises(AuthenticationError) as exc_info:
            terminology_manager.list_terminologies()
        assert "Access denied for terminology listing" in str(exc_info.value)
        
        # Throttling error
        mock_client.list_terminologies.side_effect = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'ListTerminologies'
        )
        
        with pytest.raises(RateLimitError) as exc_info:
            terminology_manager.list_terminologies()
        assert "Rate limit exceeded for terminology listing" in str(exc_info.value)
        assert exc_info.value.retry_after == 60
        
        # Generic error
        mock_client.list_terminologies.side_effect = ClientError(
            {'Error': {'Code': 'InternalError', 'Message': 'Internal error'}},
            'ListTerminologies'
        )
        
        with pytest.raises(TerminologyError) as exc_info:
            terminology_manager.list_terminologies()
        assert "Failed to list terminologies" in str(exc_info.value)
    
    def test_create_terminology_success(self, terminology_manager):
        """Test successful terminology creation."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        # Mock AWS response
        mock_response = {
            'TerminologyProperties': {
                'Arn': 'arn:123456789012:terminology/test-terminology'
            }
        }
        mock_client.import_terminology.return_value = mock_response
        
        terminology_data = TerminologyData(
            terminology_data=b"en,es\nhello,hola",
            format="CSV",
            directionality="UNI"
        )
        
        result = terminology_manager.create_terminology(
            name="test-terminology",
            description="Test terminology",
            terminology_data=terminology_data
        )
        
        assert result == 'arn:123456789012:terminology/test-terminology'
        
        mock_client.import_terminology.assert_called_once_with(
            Name="test-terminology",
            Description="Test terminology",
            TerminologyData={
                'File': b"en,es\nhello,hola",
                'Format': "CSV",
                'Directionality': "UNI"
            }
        )
    
    def test_create_terminology_with_encryption(self, terminology_manager):
        """Test terminology creation with encryption key."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_response = {
            'TerminologyProperties': {
                'Arn': 'arn:123456789012:terminology/encrypted-terminology'
            }
        }
        mock_client.import_terminology.return_value = mock_response
        
        terminology_data = TerminologyData(
            terminology_data=b"en,es\nhello,hola",
            format="CSV"
        )
        
        result = terminology_manager.create_terminology(
            name="encrypted-terminology",
            description="Encrypted terminology",
            terminology_data=terminology_data,
            encryption_key="arn:123456789012:key/12345678-1234-1234-1234-123456789012"
        )
        
        assert result == 'arn:123456789012:terminology/encrypted-terminology'
        
        expected_call = mock_client.import_terminology.call_args
        assert expected_call[1]['EncryptionKey'] == {
            'Type': 'KMS',
            'Id': 'kms'
        }
    
    def test_create_terminology_validation_errors(self, terminology_manager):
        """Test validation errors in create_terminology."""
        terminology_data = TerminologyData(
            terminology_data=b"en,es\nhello,hola",
            format="CSV"
        )
        
        # Invalid name
        with pytest.raises(ValidationError):
            terminology_manager.create_terminology("", "desc", terminology_data)
        
        with pytest.raises(ValidationError):
            terminology_manager.create_terminology("invalid name!", "desc", terminology_data)
        
        # Invalid description
        with pytest.raises(ValidationError):
            terminology_manager.create_terminology("valid-name", 123, terminology_data)
        
        # Invalid encryption key
        with pytest.raises(ValidationError):
            terminology_manager.create_terminology(
                "valid-name", "desc", terminology_data, encryption_key=123
            )
    
    def test_create_terminology_client_errors(self, terminology_manager):
        """Test client errors in create_terminology."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        terminology_data = TerminologyData(
            terminology_data=b"en,es\nhello,hola",
            format="CSV"
        )
        
        # Conflict error (terminology already exists)
        mock_client.import_terminology.side_effect = ClientError(
            {'Error': {'Code': 'ConflictException', 'Message': 'Already exists'}},
            'ImportTerminology'
        )
        
        with pytest.raises(TerminologyError) as exc_info:
            terminology_manager.create_terminology("existing-term", "desc", terminology_data)
        assert "already exists" in str(exc_info.value)
        
        # Quota exceeded error
        mock_client.import_terminology.side_effect = ClientError(
            {'Error': {'Code': 'LimitExceededException', 'Message': 'Limit exceeded'}},
            'ImportTerminology'
        )
        
        with pytest.raises(QuotaExceededError) as exc_info:
            terminology_manager.create_terminology("new-term", "desc", terminology_data)
        assert "Terminology limit exceeded" in str(exc_info.value)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('builtins.open', new_callable=mock_open)
    def test_import_terminology_csv_success(self, mock_file, mock_is_file, mock_exists, terminology_manager, sample_csv_content):
        """Test successful CSV terminology import."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = sample_csv_content
        
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_response = {
            'TerminologyProperties': {
                'Arn': 'imported-csv'
            }
        }
        mock_client.import_terminology.return_value = mock_response
        
        result = terminology_manager.import_terminology(
            name="imported-csv",
            file_path="/path/to/terminology.csv",
            description="Imported CSV terminology"
        )
        
        assert result == 'arn:123456789012:terminology/imported-csv'
        
        # Verify the import_terminology call
        call_args = mock_client.import_terminology.call_args
        assert call_args[1]['Name'] == "imported-csv"
        assert call_args[1]['Description'] == "Imported CSV terminology"
        assert call_args[1]['TerminologyData']['Format'] == "CSV"
        assert call_args[1]['TerminologyData']['File'] == sample_csv_content
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('builtins.open', new_callable=mock_open)
    def test_import_terminology_tmx_success(self, mock_file, mock_is_file, mock_exists, terminology_manager, sample_tmx_content):
        """Test successful TMX terminology import."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = sample_tmx_content
        
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_response = {
            'TerminologyProperties': {
                'Arn': 'arn:123456789012:terminology/imported-tmx'
            }
        }
        mock_client.import_terminology.return_value = mock_response
        
        result = terminology_manager.import_terminology(
            name="imported-tmx",
            file_path="/path/to/terminology.tmx",
            source_language="en",
            target_languages=["es"]
        )
        
        assert result == 'arn:123456789012:terminology/imported-tmx'
        
        # Verify the import_terminology call
        call_args = mock_client.import_terminology.call_args
        assert call_args[1]['TerminologyData']['Format'] == "TMX"
    
    def test_import_terminology_file_not_found(self, terminology_manager):
        """Test import_terminology with non-existent file."""
        with pytest.raises(FileNotFoundError):
            terminology_manager.import_terminology(
                name="test",
                file_path="/nonexistent/file.csv"
            )
    
    @patch('pathlib.Path.exists')
    def test_import_terminology_validation_errors(self, mock_exists, terminology_manager):
        """Test validation errors in import_terminology."""
        mock_exists.return_value = True
        
        # Empty name
        with pytest.raises(ValidationError):
            terminology_manager.import_terminology("", "/path/to/file.csv")
        
        # Empty file path
        with pytest.raises(ValidationError):
            terminology_manager.import_terminology("test", "")
    
    def test_get_terminology_success(self, terminology_manager):
        """Test successful terminology retrieval."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_response = {
            'TerminologyProperties': {
                'Name': 'medical-terms',
                'Description': 'Medical terminology',
                'SourceLanguageCode': 'en',
                'TargetLanguageCodes': ['es', 'fr'],
                'TermCount': 100,
                'CreatedAt': datetime(2023, 1, 1),
                'LastUpdatedAt': datetime(2023, 1, 15),
                'SizeBytes': 1024,
                'Format': 'CSV'
            }
        }
        mock_client.get_terminology.return_value = mock_response
        
        result = terminology_manager.get_terminology("medical-terms")
        
        assert isinstance(result, TerminologyDetails)
        assert result.name == 'medical-terms'
        assert result.description == 'Medical terminology'
        assert result.source_language == 'en'
        assert result.target_languages == ['es', 'fr']
        assert result.term_count == 100
        assert result.size_bytes == 1024
        assert result.format == 'CSV'
        
        mock_client.get_terminology.assert_called_once_with(
            Name="medical-terms",
            TerminologyDataFormat="CSV"
        )
    
    def test_get_terminology_not_found(self, terminology_manager):
        """Test get_terminology with non-existent terminology."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_client.get_terminology.side_effect = ClientError(
            {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Not found'}},
            'GetTerminology'
        )
        
        with pytest.raises(TerminologyError) as exc_info:
            terminology_manager.get_terminology("nonexistent")
        assert "not found" in str(exc_info.value)
    
    def test_get_terminology_validation_errors(self, terminology_manager):
        """Test validation errors in get_terminology."""
        # Invalid name
        with pytest.raises(ValidationError):
            terminology_manager.get_terminology("")
        
        # Invalid format
        with pytest.raises(ValidationError):
            terminology_manager.get_terminology("test", terminology_data_format="INVALID")
    
    def test_delete_terminology_success(self, terminology_manager):
        """Test successful terminology deletion."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        result = terminology_manager.delete_terminology("test-terminology")
        
        assert result is True
        mock_client.delete_terminology.assert_called_once_with(Name="test-terminology")
    
    def test_delete_terminology_not_found(self, terminology_manager):
        """Test delete_terminology with non-existent terminology."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_client.delete_terminology.side_effect = ClientError(
            {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Not found'}},
            'DeleteTerminology'
        )
        
        with pytest.raises(TerminologyError) as exc_info:
            terminology_manager.delete_terminology("nonexistent")
        assert "not found" in str(exc_info.value)
    
    def test_validate_terminology_conflicts(self, terminology_manager):
        """Test terminology conflict validation."""
        # Mock get_terminology calls
        def mock_get_terminology(name):
            if name == "compatible-term":
                return TerminologyDetails(
                    name="compatible-term",
                    description="Compatible",
                    source_language="en",
                    target_languages=["es", "fr"],
                    term_count=10
                )
            elif name == "incompatible-term":
                return TerminologyDetails(
                    name="incompatible-term",
                    description="Incompatible",
                    source_language="de",
                    target_languages=["it"],
                    term_count=5
                )
            else:
                raise TerminologyError(f"Terminology '{name}' not found")
        
        terminology_manager.get_terminology = Mock(side_effect=mock_get_terminology)
        
        result = terminology_manager.validate_terminology_conflicts(
            terminology_names=["compatible-term", "incompatible-term", "nonexistent-term"],
            source_language="en",
            target_language="es"
        )
        
        assert result['compatible'] == ["compatible-term"]
        assert result['incompatible'] == ["incompatible-term"]
        assert result['not_found'] == ["nonexistent-term"]
    
    def test_validate_terminology_conflicts_validation_errors(self, terminology_manager):
        """Test validation errors in validate_terminology_conflicts."""
        # Empty terminology names
        with pytest.raises(ValidationError):
            terminology_manager.validate_terminology_conflicts([], "en", "es")
        
        # Invalid source language
        with pytest.raises(ValidationError):
            terminology_manager.validate_terminology_conflicts(["test"], "invalid", "es")
        
        # Invalid target language
        with pytest.raises(ValidationError):
            terminology_manager.validate_terminology_conflicts(["test"], "en", "invalid")
    
    def test_validate_terminology_name(self, terminology_manager):
        """Test terminology name validation."""
        # Valid names
        terminology_manager._validate_terminology_name("valid-name")
        terminology_manager._validate_terminology_name("valid_name")
        terminology_manager._validate_terminology_name("ValidName123")
        
        # Invalid names
        with pytest.raises(ValidationError):
            terminology_manager._validate_terminology_name("")
        
        with pytest.raises(ValidationError):
            terminology_manager._validate_terminology_name("invalid name!")
        
        with pytest.raises(ValidationError):
            terminology_manager._validate_terminology_name("a" * 257)  # Too long
    
    def test_validate_language_code(self, terminology_manager):
        """Test language code validation."""
        # Valid codes
        terminology_manager._validate_language_code("en", "test_field")
        terminology_manager._validate_language_code("es-ES", "test_field")
        terminology_manager._validate_language_code("zh-CN", "test_field")
        
        # Invalid codes
        with pytest.raises(ValidationError):
            terminology_manager._validate_language_code("", "test_field")
        
        with pytest.raises(ValidationError):
            terminology_manager._validate_language_code("english", "test_field")
        
        with pytest.raises(ValidationError):
            terminology_manager._validate_language_code("en-us", "test_field")  # lowercase country
    
    def test_detect_file_format(self, terminology_manager):
        """Test file format detection."""
        # CSV file
        csv_path = Path("test.csv")
        csv_content = b"en,es\nhello,hola"
        assert terminology_manager._detect_file_format(csv_path, csv_content) == "CSV"
        
        # TMX file
        tmx_path = Path("test.tmx")
        tmx_content = b'<?xml version="1.0"?><tmx><body></body></tmx>'
        assert terminology_manager._detect_file_format(tmx_path, tmx_content) == "TMX"
        
        # Unknown file - defaults to CSV
        unknown_path = Path("test.txt")
        unknown_content = b"some content"
        assert terminology_manager._detect_file_format(unknown_path, unknown_content) == "CSV"
    
    def test_validate_csv_file(self, terminology_manager, sample_csv_content):
        """Test CSV file validation."""
        result = terminology_manager._validate_csv_file(sample_csv_content)
        
        assert result['source_language'] == 'en'
        assert result['target_languages'] == ['es']
        assert result['term_count'] == 3  # 3 data rows after header
    
    def test_validate_csv_file_errors(self, terminology_manager):
        """Test CSV file validation errors."""
        # Empty file
        with pytest.raises(ValidationError):
            terminology_manager._validate_csv_file(b"")
        
        # Invalid CSV
        with pytest.raises(ValidationError):
            terminology_manager._validate_csv_file(b"single_column")
        
        # Non-UTF8 content
        with pytest.raises(ValidationError):
            terminology_manager._validate_csv_file(b"\xff\xfe")
    
    def test_validate_tmx_file(self, terminology_manager, sample_tmx_content):
        """Test TMX file validation."""
        result = terminology_manager._validate_tmx_file(sample_tmx_content)
        
        assert result['source_language'] == 'en'
        assert result['target_languages'] == ['es']
        assert result['term_count'] == 2  # 2 translation units
    
    def test_validate_tmx_file_errors(self, terminology_manager):
        """Test TMX file validation errors."""
        # Invalid XML
        with pytest.raises(ValidationError):
            terminology_manager._validate_tmx_file(b"not xml")
        
        # Not TMX format
        with pytest.raises(ValidationError):
            terminology_manager._validate_tmx_file(b"<root></root>")
        
        # Missing body
        with pytest.raises(ValidationError):
            terminology_manager._validate_tmx_file(b"<tmx></tmx>")
    
    def test_service_unavailable_error(self, terminology_manager):
        """Test service unavailable error handling."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_client.list_terminologies.side_effect = BotoCoreError()
        
        with pytest.raises(ServiceUnavailableError):
            terminology_manager.list_terminologies()
    
    def test_unexpected_error(self, terminology_manager):
        """Test unexpected error handling."""
        mock_client = Mock()
        terminology_manager._translate_client = mock_client
        
        mock_client.list_terminologies.side_effect = Exception("Unexpected error")
        
        with pytest.raises(TerminologyError) as exc_info:
            terminology_manager.list_terminologies()
        assert "Unexpected error listing terminologies" in str(exc_info.value)