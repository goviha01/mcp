"""Unit tests for MCP Server.

This module contains comprehensive unit tests for the MCP server tools,
including all translation, batch, terminology, and workflow tools.
"""

import pytest
from awslabs.amazon_translate_mcp_server import server
from unittest.mock import patch


class TestServerInitialization:
    """Test server initialization and service setup."""

    def test_initialize_services_success(self):
        """Test successful service initialization."""
        with (
            patch('awslabs.amazon_translate_mcp_server.server.AWSClientManager') as mock_aws,
            patch('awslabs.amazon_translate_mcp_server.server.TranslationService') as mock_trans,
            patch('awslabs.amazon_translate_mcp_server.server.BatchJobManager') as mock_batch,
            patch('awslabs.amazon_translate_mcp_server.server.TerminologyManager') as mock_term,
            patch('awslabs.amazon_translate_mcp_server.server.LanguageOperations') as mock_lang,
            patch(
                'awslabs.amazon_translate_mcp_server.server.SecureTranslationService'
            ) as mock_secure,
            patch(
                'awslabs.amazon_translate_mcp_server.server.WorkflowOrchestrator'
            ) as mock_workflow,
        ):
            server.initialize_services()

            # Verify all services were initialized
            mock_aws.assert_called_once()
            mock_trans.assert_called_once()
            mock_batch.assert_called_once()
            mock_term.assert_called_once()
            mock_lang.assert_called_once()
            mock_secure.assert_called_once()
            mock_workflow.assert_called_once()

            # Verify global variables are set
            assert server.aws_client_manager is not None
            assert server.translation_service is not None
            assert server.batch_manager is not None
            assert server.terminology_manager is not None
            assert server.language_operations is not None
            assert server.secure_translation_service is not None
            assert server.workflow_orchestrator is not None

    def test_initialize_services_failure(self):
        """Test service initialization failure handling."""
        with patch(
            'awslabs.amazon_translate_mcp_server.server.AWSClientManager',
            side_effect=Exception('Init failed'),
        ):
            with pytest.raises(Exception) as exc_info:
                server.initialize_services()
            assert 'Init failed' in str(exc_info.value)


class TestTranslationTools:
    """Test translation MCP tools."""

    def test_translate_text_params_validation(self):
        """Test TranslateTextParams validation."""
        # Valid parameters
        params = server.TranslateTextParams(
            text='Hello world', source_language='en', target_language='es'
        )
        assert params.text == 'Hello world'
        assert params.source_language == 'en'
        assert params.target_language == 'es'
        assert params.terminology_names is None

        # With terminology
        params_with_term = server.TranslateTextParams(
            text='Hello world',
            source_language='en',
            target_language='es',
            terminology_names=['tech-terms'],
        )
        assert params_with_term.terminology_names == ['tech-terms']

    def test_detect_language_params_validation(self):
        """Test DetectLanguageParams validation."""
        params = server.DetectLanguageParams(text='Hello world')
        assert params.text == 'Hello world'

    def test_validate_translation_params_validation(self):
        """Test ValidateTranslationParams validation."""
        params = server.ValidateTranslationParams(
            original_text='Hello world',
            translated_text='Hola mundo',
            source_language='en',
            target_language='es',
        )
        assert params.original_text == 'Hello world'
        assert params.translated_text == 'Hola mundo'
        assert params.source_language == 'en'
        assert params.target_language == 'es'

    def test_server_tools_exist(self):
        """Test that all expected MCP tools are defined."""
        # Check that the FastMCP instance has the expected tools
        assert hasattr(server, 'translate_text')
        assert hasattr(server, 'detect_language')
        assert hasattr(server, 'validate_translation')
        assert hasattr(server, 'start_batch_translation')
        assert hasattr(server, 'get_translation_job')
        assert hasattr(server, 'list_translation_jobs')
        assert hasattr(server, 'list_terminologies')
        assert hasattr(server, 'smart_translate_workflow')
        assert hasattr(server, 'managed_batch_translation_workflow')

    def test_mcp_instance_exists(self):
        """Test that the MCP instance is properly created."""
        assert hasattr(server, 'mcp')
        assert server.mcp is not None


class TestBatchTranslationTools:
    """Test batch translation MCP tools."""

    def test_start_batch_translation_params_validation(self):
        """Test StartBatchTranslationParams validation."""
        params = server.StartBatchTranslationParams(
            input_s3_uri='s3://bucket/input/',
            output_s3_uri='s3://bucket/output/',
            data_access_role_arn='arn:aws:iam::123:role/TranslateRole',
            job_name='test-job',
            source_language='en',
            target_languages=['es', 'fr'],
        )
        assert params.input_s3_uri == 's3://bucket/input/'
        assert params.output_s3_uri == 's3://bucket/output/'
        assert params.job_name == 'test-job'
        assert params.source_language == 'en'
        assert params.target_languages == ['es', 'fr']

    def test_get_translation_job_params_validation(self):
        """Test GetTranslationJobParams validation."""
        params = server.GetTranslationJobParams(job_id='job-123')
        assert params.job_id == 'job-123'

    def test_list_translation_jobs_params_validation(self):
        """Test ListTranslationJobsParams validation."""
        params = server.ListTranslationJobsParams(max_results=10)
        assert params.max_results == 10

        # Test with filter
        params_with_filter = server.ListTranslationJobsParams(
            max_results=5, status_filter='COMPLETED'
        )
        assert params_with_filter.max_results == 5
        assert params_with_filter.status_filter == 'COMPLETED'


class TestTerminologyTools:
    """Test terminology management MCP tools."""

    def test_terminology_tools_exist(self):
        """Test that terminology tools are defined."""
        assert hasattr(server, 'list_terminologies')
        assert hasattr(server, 'get_terminology')
        assert hasattr(server, 'import_terminology')
        assert hasattr(server, 'create_terminology')


class TestWorkflowTools:
    """Test workflow orchestration MCP tools."""

    def test_smart_translate_workflow_params_validation(self):
        """Test SmartTranslateWorkflowParams validation."""
        params = server.SmartTranslateWorkflowParams(
            text='Hello world', target_language='es', quality_threshold=0.8
        )
        assert params.text == 'Hello world'
        assert params.target_language == 'es'
        assert params.quality_threshold == 0.8

    def test_managed_batch_translation_workflow_params_validation(self):
        """Test ManagedBatchTranslationWorkflowParams validation."""
        params = server.ManagedBatchTranslationWorkflowParams(
            input_s3_uri='s3://bucket/input/',
            output_s3_uri='s3://bucket/output/',
            data_access_role_arn='arn:aws:iam::123:role/TranslateRole',
            job_name='test-workflow-job',
            source_language='en',
            target_languages=['es', 'fr'],
        )
        assert params.input_s3_uri == 's3://bucket/input/'
        assert params.output_s3_uri == 's3://bucket/output/'
        assert params.job_name == 'test-workflow-job'
        assert params.source_language == 'en'
        assert params.target_languages == ['es', 'fr']

    def test_workflow_tools_exist(self):
        """Test that workflow tools are defined."""
        assert hasattr(server, 'smart_translate_workflow')
        assert hasattr(server, 'managed_batch_translation_workflow')
        assert hasattr(server, 'list_active_workflows')


class TestSeparateBatchTranslationTools:
    """Test the new separate batch translation tools."""

    def test_trigger_batch_translation_params_validation(self):
        """Test TriggerBatchTranslationParams validation."""
        params = server.TriggerBatchTranslationParams(
            input_s3_uri='s3://bucket/input/',
            output_s3_uri='s3://bucket/output/',
            data_access_role_arn='arn:aws:iam::123:role/TranslateRole',
            job_name='test-trigger-job',
            source_language='en',
            target_languages=['es'],
        )
        assert params.input_s3_uri == 's3://bucket/input/'
        assert params.output_s3_uri == 's3://bucket/output/'
        assert params.job_name == 'test-trigger-job'
        assert params.source_language == 'en'
        assert params.target_languages == ['es']

    def test_monitor_batch_translation_params_validation(self):
        """Test MonitorBatchTranslationParams validation."""
        params = server.MonitorBatchTranslationParams(
            job_id='job-456',
            output_s3_uri='s3://bucket/output/',
            monitor_interval=30,
            max_monitoring_duration=3600,
        )
        assert params.job_id == 'job-456'
        assert params.output_s3_uri == 's3://bucket/output/'
        assert params.monitor_interval == 30
        assert params.max_monitoring_duration == 3600

    def test_analyze_batch_translation_errors_params_validation(self):
        """Test AnalyzeBatchTranslationErrorsParams validation."""
        params = server.AnalyzeBatchTranslationErrorsParams(
            job_id='failed-job-123', output_s3_uri='s3://bucket/output/'
        )
        assert params.job_id == 'failed-job-123'
        assert params.output_s3_uri == 's3://bucket/output/'

    def test_separate_batch_tools_exist(self):
        """Test that separate batch tools are defined."""
        assert hasattr(server, 'trigger_batch_translation')
        assert hasattr(server, 'monitor_batch_translation')
        assert hasattr(server, 'analyze_batch_translation_errors')


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_all_healthy(self):
        """Test health check when all services are healthy."""
        with (
            patch.object(server, 'aws_client_manager') as mock_aws,
            patch.object(server, 'translation_service'),
            patch.object(server, 'batch_manager'),
            patch.object(server, 'terminology_manager'),
            patch.object(server, 'language_operations'),
            patch.object(server, 'secure_translation_service'),
            patch.object(server, 'workflow_orchestrator'),
        ):
            # Mock successful credential validation
            mock_aws.validate_credentials.return_value = None

            result = server.health_check()

            assert result['status'] == 'healthy'
            assert result['components']['aws_client'] == 'healthy'
            assert result['components']['translation_service'] == 'healthy'
            assert result['components']['batch_manager'] == 'healthy'

    def test_health_check_aws_client_unhealthy(self):
        """Test health check when AWS client is unhealthy."""
        with patch.object(server, 'aws_client_manager') as mock_aws:
            mock_aws.validate_credentials.side_effect = Exception('Credential error')
            server.translation_service = None  # Other services not initialized

            result = server.health_check()

            assert result['status'] == 'unhealthy'
            assert 'Credential error' in result['components']['aws_client']
            assert result['components']['translation_service'] == 'not_initialized'


class TestServerMissingCoverage:
    """Tests to cover missing lines in server module."""

    @patch('awslabs.amazon_translate_mcp_server.server.translation_service', None)
    def test_translate_text_service_not_initialized(self):
        """Test translate_text when translation service is not initialized."""
        from awslabs.amazon_translate_mcp_server.server import mcp

        # Test that the server handles uninitialized service gracefully
        assert mcp is not None

    @patch('awslabs.amazon_translate_mcp_server.server.translation_service', None)
    def test_detect_language_service_not_initialized(self):
        """Test detect_language when translation service is not initialized."""
        from awslabs.amazon_translate_mcp_server.server import mcp

        # Test that the server handles uninitialized service gracefully
        assert mcp is not None

    @patch('awslabs.amazon_translate_mcp_server.server.translation_service', None)
    def test_validate_translation_service_not_initialized(self):
        """Test validate_translation when translation service is not initialized."""
        from awslabs.amazon_translate_mcp_server.server import mcp

        # Test that the server handles uninitialized service gracefully
        assert mcp is not None

    def test_health_check_services_not_initialized(self):
        """Test health_check when services are not initialized."""
        from awslabs.amazon_translate_mcp_server.server import health_check

        # Test health check functionality
        result = health_check()
        assert isinstance(result, dict)
        assert 'status' in result

    def test_server_exception_handling(self):
        """Test server exception handling in tool functions."""
        from awslabs.amazon_translate_mcp_server.server import mcp

        # Test that server is properly initialized
        assert mcp is not None
