# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests to cover remaining missing lines and achieve near 100% coverage.

This module contains targeted tests for specific edge cases and error paths
that are not covered by the main test suites.
"""

import pytest
from awslabs.amazon_translate_mcp_server import server
from awslabs.amazon_translate_mcp_server.exceptions import (
    SecurityError,
    TranslateException,
    ValidationError,
    map_aws_error,
)
from botocore.exceptions import ClientError
from unittest.mock import MagicMock, patch


class TestServerClientErrorHandling:
    """Test server.py ClientError handling paths (lines 300-302)."""

    @pytest.mark.asyncio
    async def test_translate_text_with_client_error(self):
        """Test translate_text handling of AWS ClientError."""
        mock_ctx = MagicMock()

        with patch.object(server, 'translation_service') as mock_service:
            # Create a ClientError
            error = ClientError(
                {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                'TranslateText',
            )
            mock_service.translate_text.side_effect = error

            result = await server.translate_text(
                ctx=mock_ctx, text='Hello world', source_language='en', target_language='es'
            )

            # Should return normalized error response
            assert 'error' in result
            assert 'error_type' in result
            assert 'error_code' in result
            assert 'correlation_id' in result

    @pytest.mark.asyncio
    async def test_detect_language_with_client_error(self):
        """Test detect_language handling of AWS ClientError."""
        mock_ctx = MagicMock()

        with patch.object(server, 'translation_service') as mock_service:
            error = ClientError(
                {'Error': {'Code': 'InvalidRequestException', 'Message': 'Invalid request'}},
                'DetectDominantLanguage',
            )
            mock_service.detect_language.side_effect = error

            result = await server.detect_language(ctx=mock_ctx, text='Hello')

            assert 'error' in result
            assert 'correlation_id' in result


class TestServerTranslateExceptionHandling:
    """Test server.py TranslateException handling paths (lines 287-295)."""

    @pytest.mark.asyncio
    async def test_translate_text_with_translate_exception(self):
        """Test translate_text handling of TranslateException."""
        mock_ctx = MagicMock()

        with patch.object(server, 'translation_service') as mock_service:
            # Create a TranslateException
            exc = TranslateException(
                message='Custom translation error',
                error_code='CUSTOM_ERROR',
                details={'key': 'value'},
                correlation_id='test-123',
            )
            mock_service.translate_text.side_effect = exc

            result = await server.translate_text(
                ctx=mock_ctx, text='Hello world', source_language='en', target_language='es'
            )

            # Should return the TranslateException's error response
            assert 'error' in result
            assert result['error_type'] == 'TranslateException'
            assert result['correlation_id'] == 'test-123'
            assert 'details' in result


class TestServerBatchTranslationEdgeCases:
    """Test batch translation edge cases in server.py."""

    @pytest.mark.asyncio
    async def test_start_batch_translation_with_validation_error(self):
        """Test start_batch_translation with validation errors."""
        mock_ctx = MagicMock()

        # Test with invalid S3 URI
        result = await server.start_batch_translation(
            ctx=mock_ctx,
            input_s3_uri='invalid-uri',
            output_s3_uri='s3://bucket/output/',
            data_access_role_arn='arn:aws:iam::123456789012:role/Role',
            job_name='test-job',
            source_language='en',
            target_languages=['es'],
        )

        assert 'error' in result

    @pytest.mark.asyncio
    async def test_get_translation_job_not_initialized(self):
        """Test get_translation_job when batch_manager is None."""
        mock_ctx = MagicMock()

        with patch.object(server, 'batch_manager', None):
            result = await server.get_translation_job(ctx=mock_ctx, job_id='test-job-123')

            assert 'error' in result


class TestServerWorkflowEdgeCases:
    """Test workflow edge cases in server.py."""

    @pytest.mark.asyncio
    async def test_smart_translate_workflow_not_initialized(self):
        """Test smart_translate_workflow when orchestrator is None."""
        mock_ctx = MagicMock()

        with patch.object(server, 'workflow_orchestrator', None):
            result = await server.smart_translate_workflow(
                ctx=mock_ctx, text='Hello world', target_language='es'
            )

            assert 'error' in result

    @pytest.mark.asyncio
    async def test_list_active_workflows_not_initialized(self):
        """Test list_active_workflows when orchestrator is None."""
        mock_ctx = MagicMock()

        with patch.object(server, 'workflow_orchestrator', None):
            result = await server.list_active_workflows(ctx=mock_ctx)

            assert 'error' in result


class TestServerTerminologyEdgeCases:
    """Test terminology edge cases in server.py."""

    @pytest.mark.asyncio
    async def test_list_terminologies_not_initialized(self):
        """Test list_terminologies when manager is None."""
        mock_ctx = MagicMock()

        with patch.object(server, 'terminology_manager', None):
            result = await server.list_terminologies(ctx=mock_ctx)

            assert 'error' in result

    @pytest.mark.asyncio
    async def test_import_terminology_with_invalid_base64(self):
        """Test import_terminology with invalid base64 content."""
        mock_ctx = MagicMock()

        result = await server.import_terminology(
            ctx=mock_ctx,
            name='test-term',
            file_content_base64='invalid-base64!!!',
            file_format='CSV',
        )

        assert 'error' in result


class TestServerLanguageOperationsEdgeCases:
    """Test language operations edge cases in server.py."""

    @pytest.mark.asyncio
    async def test_list_language_pairs_not_initialized(self):
        """Test list_language_pairs when operations is None."""
        mock_ctx = MagicMock()

        with patch.object(server, 'language_operations', None):
            result = await server.list_language_pairs(ctx=mock_ctx)

            assert 'error' in result

    @pytest.mark.asyncio
    async def test_get_language_metrics_not_initialized(self):
        """Test get_language_metrics when operations is None."""
        mock_ctx = MagicMock()

        with patch.object(server, 'language_operations', None):
            result = await server.get_language_metrics(ctx=mock_ctx, language_code='en')

            assert 'error' in result


class TestTerminologyManagerMissingLines:
    """Test terminology_manager.py missing lines."""

    def test_create_terminology_with_client_error(self):
        """Test create_terminology with AWS ClientError."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.terminology_manager import TerminologyManager

        client_manager = AWSClientManager()
        manager = TerminologyManager(client_manager)

        with patch.object(manager.translate_client, 'import_terminology') as mock_import:
            error = ClientError(
                {'Error': {'Code': 'LimitExceededException', 'Message': 'Limit exceeded'}},
                'ImportTerminology',
            )
            mock_import.side_effect = error

            with pytest.raises(Exception):
                manager.create_terminology(
                    name='test-term',
                    source_language='en',
                    target_language='es',
                    terms=[{'source': 'hello', 'target': 'hola'}],
                )

    def test_delete_terminology_with_client_error(self):
        """Test delete_terminology with AWS ClientError."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.terminology_manager import TerminologyManager

        client_manager = AWSClientManager()
        manager = TerminologyManager(client_manager)

        with patch.object(manager.translate_client, 'delete_terminology') as mock_delete:
            error = ClientError(
                {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Not found'}},
                'DeleteTerminology',
            )
            mock_delete.side_effect = error

            with pytest.raises(Exception):
                manager.delete_terminology('nonexistent-term')


class TestBatchManagerMissingLines:
    """Test batch_manager.py missing lines."""

    def test_cancel_job_with_client_error(self):
        """Test cancel_job with AWS ClientError."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.batch_manager import BatchTranslationManager

        client_manager = AWSClientManager()
        manager = BatchTranslationManager(client_manager)

        with patch.object(manager.translate_client, 'stop_text_translation_job') as mock_stop:
            error = ClientError(
                {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Job not found'}},
                'StopTextTranslationJob',
            )
            mock_stop.side_effect = error

            with pytest.raises(Exception):
                manager.cancel_job('nonexistent-job')

    def test_list_jobs_with_filters(self):
        """Test list_jobs with status filter."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.batch_manager import BatchTranslationManager

        client_manager = AWSClientManager()
        manager = BatchTranslationManager(client_manager)

        with patch.object(manager.translate_client, 'list_text_translation_jobs') as mock_list:
            mock_list.return_value = {'TextTranslationJobPropertiesList': []}

            result = manager.list_jobs(status_filter='COMPLETED')
            assert result == []


class TestWorkflowOrchestratorMissingLines:
    """Test workflow_orchestrator.py missing lines."""

    @pytest.mark.asyncio
    async def test_smart_translate_with_terminology(self):
        """Test smart_translate_workflow with terminology."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.workflow_orchestrator import WorkflowOrchestrator

        client_manager = AWSClientManager()
        orchestrator = WorkflowOrchestrator(client_manager)

        with patch.object(orchestrator.translation_service, 'detect_language') as mock_detect:
            mock_detect.return_value = MagicMock(language_code='en', confidence=0.99)

            with patch.object(
                orchestrator.translation_service, 'translate_text'
            ) as mock_translate:
                mock_translate.return_value = 'Hola mundo'

                result = await orchestrator.smart_translate_workflow(
                    text='Hello world', target_language='es', terminology_names=['tech-terms']
                )

                assert result.translated_text == 'Hola mundo'

    @pytest.mark.asyncio
    async def test_get_workflow_status_not_found(self):
        """Test get_workflow_status for non-existent workflow."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.exceptions import WorkflowError
        from awslabs.amazon_translate_mcp_server.workflow_orchestrator import WorkflowOrchestrator

        client_manager = AWSClientManager()
        orchestrator = WorkflowOrchestrator(client_manager)

        with pytest.raises(WorkflowError):
            await orchestrator.get_workflow_status('nonexistent-workflow-id')


class TestLanguageOperationsMissingLines:
    """Test language_operations.py missing lines."""

    def test_get_supported_languages_with_cache_miss(self):
        """Test get_supported_languages with cache miss."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.language_operations import LanguageOperations

        client_manager = AWSClientManager()
        ops = LanguageOperations(client_manager, enable_cache=True)

        with patch.object(ops.translate_client, 'list_languages') as mock_list:
            mock_list.return_value = {
                'Languages': [
                    {'LanguageCode': 'en', 'LanguageName': 'English'},
                    {'LanguageCode': 'es', 'LanguageName': 'Spanish'},
                ]
            }

            result = ops.get_supported_languages()
            assert len(result) >= 2

    def test_validate_language_pair_invalid(self):
        """Test validate_language_pair with invalid pair."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.language_operations import LanguageOperations

        client_manager = AWSClientManager()
        ops = LanguageOperations(client_manager)

        # Test with same source and target
        result = ops.validate_language_pair('en', 'en')
        assert result is False


class TestSecurityValidatorsMissingLines:
    """Test security_validators.py missing lines."""

    def test_validate_text_input_with_none(self):
        """Test validate_text_input with None value."""
        from awslabs.amazon_translate_mcp_server.security_validators import validate_text_input

        with pytest.raises(ValidationError):
            validate_text_input(None)

    def test_validate_s3_uri_with_localhost(self):
        """Test validate_s3_uri blocks localhost."""
        from awslabs.amazon_translate_mcp_server.security_validators import validate_s3_uri

        with pytest.raises(SecurityError):
            validate_s3_uri('s3://localhost/bucket/key')

    def test_validate_target_languages_empty_list(self):
        """Test validate_target_languages with empty list."""
        from awslabs.amazon_translate_mcp_server.security_validators import (
            validate_target_languages,
        )

        with pytest.raises(ValidationError):
            validate_target_languages([])


class TestConfigMissingLines:
    """Test config.py missing lines."""

    def test_server_config_with_invalid_values(self):
        """Test ServerConfig with invalid values."""
        from awslabs.amazon_translate_mcp_server.config import ServerConfig

        # Test with negative max_text_length
        with pytest.raises(ValueError):
            ServerConfig(max_text_length=-1)

    def test_validate_aws_config_with_invalid_credentials(self):
        """Test validate_aws_config with invalid credentials."""
        from awslabs.amazon_translate_mcp_server.config import ServerConfig, validate_aws_config

        config = ServerConfig(
            aws_region='us-east-1',
            aws_access_key_id='',  # Empty key
            aws_secret_access_key='',  # Empty secret
        )

        # Should still validate (credentials are optional)
        result = validate_aws_config(config)
        assert result is True


class TestExceptionsMissingLines:
    """Test exceptions.py missing lines."""

    def test_translate_exception_with_all_params(self):
        """Test TranslateException with all parameters."""
        exc = TranslateException(
            message='Test error',
            error_code='TEST_CODE',
            details={'key': 'value'},
            correlation_id='test-123',
            retry_after=30,
        )

        response = exc.to_error_response()
        assert response.message == 'Test error'
        assert response.error_code == 'TEST_CODE'
        assert response.retry_after == 30

    def test_map_aws_error_with_access_denied(self):
        """Test map_aws_error with AccessDeniedException."""
        error = ClientError(
            {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}},
            'TestOperation',
        )

        result = map_aws_error(error, 'test-id')
        from awslabs.amazon_translate_mcp_server.exceptions import AuthenticationError

        assert isinstance(result, AuthenticationError)


class TestRetryHandlerMissingLines:
    """Test retry_handler.py missing lines."""

    def test_retry_handler_with_non_retryable_error(self):
        """Test retry handler with non-retryable error."""
        from awslabs.amazon_translate_mcp_server.retry_handler import RetryHandler

        handler = RetryHandler()

        def failing_function():
            raise ValueError('Non-retryable error')

        with pytest.raises(ValueError):
            handler.execute_with_retry(failing_function)


class TestLoggingConfigMissingLines:
    """Test logging_config.py missing lines."""

    def test_setup_logging_with_file_handler(self):
        """Test setup_logging with file output."""
        import os
        import tempfile
        from awslabs.amazon_translate_mcp_server.logging_config import setup_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'test.log')
            setup_logging(log_level='INFO', log_file=log_file)

            # Verify log file was created
            assert os.path.exists(log_file)


class TestTranslationServiceMissingLines:
    """Test translation_service.py missing lines."""

    def test_validate_translation_with_low_quality(self):
        """Test validate_translation with low quality score."""
        from awslabs.amazon_translate_mcp_server.aws_client import AWSClientManager
        from awslabs.amazon_translate_mcp_server.translation_service import TranslationService

        client_manager = AWSClientManager()
        service = TranslationService(client_manager)

        # Mock a low quality translation
        with patch.object(service, '_calculate_quality_score') as mock_quality:
            mock_quality.return_value = 0.3  # Low quality

            result = service.validate_translation(
                original_text='Hello world',
                translated_text='Bad translation',
                source_language='en',
                target_language='es',
            )

            assert result.quality_score < 0.5
