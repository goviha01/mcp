#!/usr/bin/env python3
"""
Amazon Translate MCP Server.

A Model Context Protocol server that provides AI assistants with comprehensive access
to Amazon Translate services including real-time translation, batch processing,
terminology management, and language operations.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .aws_client import AWSClientManager
from .translation_service import TranslationService
from .batch_manager import BatchJobManager
from .terminology_manager import TerminologyManager
from .language_operations import LanguageOperations
from .secure_translation_service import SecureTranslationService
from .workflow_orchestrator import WorkflowOrchestrator
from .config import ServerConfig, load_config_from_env
from .logging_config import setup_logging
from .exceptions import (
    TranslateException,
    AuthenticationError,
    ValidationError,
    TranslationError,
    TerminologyError,
    BatchJobError,
    WorkflowError,
)


# Initialize logging
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Amazon Translate MCP Server")


# Pydantic models for tool parameters
class TranslateTextParams(BaseModel):
    """Parameters for translate_text tool."""
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="Source language code (e.g., 'en', 'es', 'fr')")
    target_language: str = Field(..., description="Target language code (e.g., 'en', 'es', 'fr')")
    terminology_names: Optional[List[str]] = Field(
        default=None, description="List of custom terminology names to apply"
    )


class DetectLanguageParams(BaseModel):
    """Parameters for detect_language tool."""
    text: str = Field(..., description="Text to analyze for language detection")


class ValidateTranslationParams(BaseModel):
    """Parameters for validate_translation tool."""
    original_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text to validate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")


class StartBatchTranslationParams(BaseModel):
    """Parameters for start_batch_translation tool."""
    input_s3_uri: str = Field(..., description="S3 URI for input documents")
    output_s3_uri: str = Field(..., description="S3 URI for output location")
    data_access_role_arn: str = Field(..., description="IAM role ARN for S3 access")
    job_name: str = Field(..., description="Name for the translation job")
    source_language: str = Field(..., description="Source language code")
    target_languages: List[str] = Field(..., description="List of target language codes")
    content_type: str = Field(default="text/plain", description="Content type of input documents")
    terminology_names: Optional[List[str]] = Field(
        default=None, description="List of custom terminology names to apply"
    )


class GetTranslationJobParams(BaseModel):
    """Parameters for get_translation_job tool."""
    job_id: str = Field(..., description="Translation job ID")


class ListTranslationJobsParams(BaseModel):
    """Parameters for list_translation_jobs tool."""
    status_filter: Optional[str] = Field(
        default=None, description="Filter jobs by status (SUBMITTED, IN_PROGRESS, COMPLETED, FAILED, STOPPED)"
    )
    max_results: int = Field(default=50, description="Maximum number of jobs to return")


class CreateTerminologyParams(BaseModel):
    """Parameters for create_terminology tool."""
    name: str = Field(..., description="Name for the terminology")
    description: str = Field(..., description="Description of the terminology")
    source_language: str = Field(..., description="Source language code")
    target_languages: List[str] = Field(..., description="List of target language codes")
    terms: List[Dict[str, str]] = Field(..., description="List of term pairs")


class ImportTerminologyParams(BaseModel):
    """Parameters for import_terminology tool."""
    name: str = Field(..., description="Name for the terminology")
    description: str = Field(..., description="Description of the terminology")
    file_content: str = Field(..., description="Base64 encoded terminology file content")
    file_format: str = Field(..., description="File format (CSV or TMX)")
    source_language: str = Field(..., description="Source language code")
    target_languages: List[str] = Field(..., description="List of target language codes")


class GetTerminologyParams(BaseModel):
    """Parameters for get_terminology tool."""
    name: str = Field(..., description="Name of the terminology")


class GetLanguageMetricsParams(BaseModel):
    """Parameters for get_language_metrics tool."""
    language_pair: Optional[str] = Field(
        default=None, description="Language pair (e.g., 'en-es') to get metrics for"
    )
    time_range: str = Field(default="24h", description="Time range for metrics (24h, 7d, 30d)")


# Workflow Parameter Models

class SmartTranslateWorkflowParams(BaseModel):
    """Parameters for smart_translate_workflow tool."""
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code (e.g., 'en', 'es', 'fr')")
    quality_threshold: float = Field(default=0.8, description="Minimum quality score threshold (0.0-1.0)")
    terminology_names: Optional[List[str]] = Field(
        default=None, description="List of custom terminology names to apply"
    )
    auto_detect_language: bool = Field(default=True, description="Whether to auto-detect source language")


class ManagedBatchTranslationWorkflowParams(BaseModel):
    """Parameters for managed_batch_translation_workflow tool."""
    input_s3_uri: str = Field(..., description="S3 URI for input documents")
    output_s3_uri: str = Field(..., description="S3 URI for output location")
    data_access_role_arn: str = Field(..., description="IAM role ARN for S3 access")
    job_name: str = Field(..., description="Name for the translation job")
    source_language: str = Field(..., description="Source language code")
    target_languages: List[str] = Field(..., description="List of target language codes")
    terminology_names: Optional[List[str]] = Field(
        default=None, description="List of custom terminology names to apply"
    )
    content_type: str = Field(default="text/plain", description="Content type of input documents")
    monitor_interval: int = Field(default=30, description="Monitoring interval in seconds")
    max_monitoring_duration: int = Field(default=3600, description="Maximum monitoring duration in seconds")


# Global service instances
aws_client_manager: Optional[AWSClientManager] = None
translation_service: Optional[TranslationService] = None
secure_translation_service: Optional[SecureTranslationService] = None
batch_manager: Optional[BatchJobManager] = None
terminology_manager: Optional[TerminologyManager] = None
language_operations: Optional[LanguageOperations] = None
workflow_orchestrator: Optional[WorkflowOrchestrator] = None


def initialize_services() -> None:
    """Initialize all service components."""
    global aws_client_manager, translation_service, secure_translation_service
    global batch_manager, terminology_manager, language_operations, workflow_orchestrator
    
    try:
        logger.info("Initializing Amazon Translate MCP Server services...")
        
        # Initialize AWS client manager
        aws_client_manager = AWSClientManager()
        
        # Initialize core services
        translation_service = TranslationService(aws_client_manager)
        secure_translation_service = SecureTranslationService(translation_service)
        batch_manager = BatchJobManager(aws_client_manager)
        terminology_manager = TerminologyManager(aws_client_manager)
        language_operations = LanguageOperations(aws_client_manager)
        
        # Initialize workflow orchestrator
        workflow_orchestrator = WorkflowOrchestrator(
            translation_service=translation_service,
            batch_manager=batch_manager,
            terminology_manager=terminology_manager,
            language_operations=language_operations
        )
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@mcp.tool()
async def translate_text(params: TranslateTextParams) -> Dict[str, Any]:
    """
    Translate text from one language to another using Amazon Translate.
    
    This tool provides real-time text translation with support for custom terminology
    and automatic language detection.
    """
    try:
        if not secure_translation_service:
            raise TranslationError("Translation service not initialized")
            
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            secure_translation_service.translate_text,
            params.text,
            params.source_language,
            params.target_language,
            params.terminology_names or []
        )
        
        return {
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "applied_terminologies": result.applied_terminologies,
            "confidence_score": result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def detect_language(params: DetectLanguageParams) -> Dict[str, Any]:
    """
    Detect the language of the provided text using Amazon Translate.
    
    Returns the detected language with confidence score and alternative language candidates.
    """
    try:
        if not translation_service:
            raise TranslationError("Translation service not initialized")
            
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            translation_service.detect_language,
            params.text
        )
        
        return {
            "detected_language": result.detected_language,
            "confidence_score": result.confidence_score,
            "alternative_languages": result.alternative_languages
        }
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def validate_translation(params: ValidateTranslationParams) -> Dict[str, Any]:
    """
    Validate the quality of a translation using various quality checks.
    
    Performs quality assessment and provides suggestions for improvement.
    """
    try:
        if not translation_service:
            raise TranslationError("Translation service not initialized")
            
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            translation_service.validate_translation,
            params.original_text,
            params.translated_text,
            params.source_language,
            params.target_language
        )
        
        return {
            "is_valid": result.is_valid,
            "quality_score": result.quality_score,
            "issues": result.issues,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"Translation validation failed: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def start_batch_translation(params: StartBatchTranslationParams) -> Dict[str, Any]:
    """
    Start a batch translation job for processing multiple documents.
    
    Supports various document formats and custom terminology application.
    """
    try:
        if not batch_manager:
            raise BatchJobError("Batch manager not initialized")
        
        # Import the required models
        from .models import BatchInputConfig, BatchOutputConfig, JobConfig
        
        # Create configuration objects
        input_config = BatchInputConfig(
            s3_uri=params.input_s3_uri,
            content_type=params.content_type,
            data_access_role_arn=params.data_access_role_arn
        )
        
        output_config = BatchOutputConfig(
            s3_uri=params.output_s3_uri,
            data_access_role_arn=params.data_access_role_arn
        )
        
        job_config = JobConfig(
            job_name=params.job_name,
            source_language_code=params.source_language,
            target_language_codes=params.target_languages,
            terminology_names=params.terminology_names or []
        )
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        job_id = await loop.run_in_executor(
            None,
            batch_manager.start_batch_translation,
            input_config,
            output_config,
            job_config
        )
        
        return {
            "job_id": job_id,
            "status": "SUBMITTED",
            "message": "Batch translation job started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start batch translation: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def get_translation_job(params: GetTranslationJobParams) -> Dict[str, Any]:
    """
    Get the status and details of a translation job.
    
    Returns current job status, progress, and results location when completed.
    """
    try:
        if not batch_manager:
            raise BatchJobError("Batch manager not initialized")
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        job_status = await loop.run_in_executor(
            None,
            batch_manager.get_translation_job,
            params.job_id
        )
        
        return {
            "job_id": job_status.job_id,
            "job_name": job_status.job_name,
            "status": job_status.status.value,
            "progress": job_status.progress,
            "input_config": {
                "s3_uri": job_status.input_config.s3_uri if job_status.input_config else None,
                "content_type": job_status.input_config.content_type if job_status.input_config else None
            },
            "output_config": {
                "s3_uri": job_status.output_config.s3_uri if job_status.output_config else None
            },
            "created_at": job_status.created_at.isoformat() if job_status.created_at else None,
            "completed_at": job_status.completed_at.isoformat() if job_status.completed_at else None,
            "error_details": job_status.error_details
        }
        
    except Exception as e:
        logger.error(f"Failed to get translation job: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def list_translation_jobs(params: ListTranslationJobsParams) -> Dict[str, Any]:
    """
    List translation jobs with optional status filtering.
    
    Returns a list of jobs with their current status and metadata.
    """
    try:
        if not batch_manager:
            raise BatchJobError("Batch manager not initialized")
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        jobs = await loop.run_in_executor(
            None,
            batch_manager.list_translation_jobs,
            params.status_filter,
            params.max_results
        )
        
        job_list = []
        for job in jobs:
            job_list.append({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status.value,
                "source_language": job.source_language,
                "target_languages": job.target_languages,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            })
        
        return {
            "jobs": job_list,
            "total_count": len(job_list)
        }
        
    except Exception as e:
        logger.error(f"Failed to list translation jobs: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def list_terminologies() -> Dict[str, Any]:
    """
    List all available custom terminologies.
    
    Returns a list of terminologies with their metadata and language pairs.
    """
    try:
        if not terminology_manager:
            raise TerminologyError("Terminology manager not initialized")
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            terminology_manager.list_terminologies
        )
        
        # The method returns a dict with 'terminologies' and 'next_token'
        terminologies = result.get('terminologies', [])
        
        terminology_list = []
        for terminology in terminologies:
            terminology_list.append({
                "name": terminology.name,
                "description": terminology.description,
                "source_language": terminology.source_language,
                "target_languages": terminology.target_languages,
                "term_count": terminology.term_count,
                "created_at": terminology.created_at.isoformat() if terminology.created_at else None,
                "last_updated": terminology.last_updated.isoformat() if terminology.last_updated else None
            })
        
        return {
            "terminologies": terminology_list,
            "total_count": len(terminology_list),
            "next_token": result.get('next_token')
        }
        
    except Exception as e:
        logger.error(f"Failed to list terminologies: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def create_terminology(params: CreateTerminologyParams) -> Dict[str, Any]:
    """
    Create a new custom terminology for consistent translations.
    
    Accepts term pairs and creates a terminology that can be applied to translations.
    """
    try:
        if not terminology_manager:
            raise TerminologyError("Terminology manager not initialized")
        
        # Import the required models
        from .models import TerminologyData
        
        # Create terminology data object
        terminology_data = TerminologyData(
            format="CSV",
            file_content=b"",  # Will be populated from terms
            source_language=params.source_language,
            target_languages=params.target_languages
        )
        
        # Convert terms to CSV format
        csv_content = "source,target\n"
        for term in params.terms:
            source = term.get("source", "")
            target = term.get("target", "")
            csv_content += f'"{source}","{target}"\n'
        
        terminology_data.file_content = csv_content.encode('utf-8')
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        terminology_arn = await loop.run_in_executor(
            None,
            terminology_manager.create_terminology,
            params.name,
            params.description,
            terminology_data
        )
        
        return {
            "terminology_arn": terminology_arn,
            "name": params.name,
            "status": "CREATED",
            "message": "Terminology created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create terminology: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def import_terminology(params: ImportTerminologyParams) -> Dict[str, Any]:
    """
    Import terminology from a file (CSV or TMX format).
    
    Supports importing terminology data from external files for consistent translations.
    """
    try:
        if not terminology_manager:
            raise TerminologyError("Terminology manager not initialized")
        
        import base64
        import tempfile
        import os
        
        # Decode base64 file content
        file_content = base64.b64decode(params.file_content)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{params.file_format.lower()}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Run synchronous method in thread pool
            loop = asyncio.get_event_loop()
            terminology_arn = await loop.run_in_executor(
                None,
                terminology_manager.import_terminology,
                params.name,
                temp_file_path,
                params.description,
                params.source_language,
                params.target_languages,
                params.file_format
            )
            
            return {
                "terminology_arn": terminology_arn,
                "name": params.name,
                "status": "IMPORTED",
                "message": "Terminology imported successfully"
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logger.error(f"Failed to import terminology: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def get_terminology(params: GetTerminologyParams) -> Dict[str, Any]:
    """
    Get detailed information about a specific terminology.
    
    Returns terminology metadata, term pairs, and usage statistics.
    """
    try:
        if not terminology_manager:
            raise TerminologyError("Terminology manager not initialized")
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        terminology = await loop.run_in_executor(
            None,
            terminology_manager.get_terminology,
            params.name
        )
        
        return {
            "name": terminology.name,
            "description": terminology.description,
            "source_language": terminology.source_language,
            "target_languages": terminology.target_languages,
            "term_count": terminology.term_count,
            "created_at": terminology.created_at.isoformat() if terminology.created_at else None,
            "last_updated": terminology.last_updated.isoformat() if terminology.last_updated else None,
            "size_bytes": terminology.size_bytes,
            "format": terminology.format
        }
        
    except Exception as e:
        logger.error(f"Failed to get terminology: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def list_language_pairs() -> Dict[str, Any]:
    """
    List all supported language pairs for translation.
    
    Returns all available source-target language combinations with their capabilities.
    """
    try:
        if not language_operations:
            raise TranslationError("Language operations not initialized")
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        language_pairs = await loop.run_in_executor(
            None,
            language_operations.list_language_pairs
        )
        
        pairs_list = []
        for pair in language_pairs:
            pairs_list.append({
                "source_language": pair.source_language,
                "target_language": pair.target_language,
                "supported_formats": pair.supported_formats,
                "custom_terminology_supported": pair.custom_terminology_supported
            })
        
        return {
            "language_pairs": pairs_list,
            "total_count": len(pairs_list)
        }
        
    except Exception as e:
        logger.error(f"Failed to list language pairs: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def get_language_metrics(params: GetLanguageMetricsParams) -> Dict[str, Any]:
    """
    Get usage metrics and statistics for language operations.
    
    Returns translation volume, performance metrics, and usage patterns.
    """
    try:
        if not language_operations:
            raise TranslationError("Language operations not initialized")
        
        # Run synchronous method in thread pool
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            language_operations.get_language_metrics,
            params.language_pair,
            params.time_range
        )
        
        return {
            "language_pair": metrics.language_pair,
            "time_range": metrics.time_range,
            "translation_count": metrics.translation_count,
            "character_count": metrics.character_count,
            "average_response_time": metrics.average_response_time,
            "success_rate": metrics.success_rate,
            "error_rate": metrics.error_rate,
            "cost_estimate": metrics.cost_estimate
        }
        
    except Exception as e:
        logger.error(f"Failed to get language metrics: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


# Workflow Orchestration Tools

@mcp.tool()
async def smart_translate_workflow(params: SmartTranslateWorkflowParams) -> Dict[str, Any]:
    """
    Execute intelligent translation workflow with automatic language detection and quality validation.
    
    This workflow combines multiple translation operations into a single, intelligent process:
    1. Automatically detects source language (if enabled)
    2. Validates language pair support
    3. Translates text with optional terminology
    4. Validates translation quality
    5. Returns comprehensive results with quality metrics
    
    Benefits:
    - Eliminates manual language specification
    - Built-in quality assurance with confidence scoring
    - Automatic language pair validation
    - Comprehensive results with detection, translation, and quality metrics
    """
    try:
        if not workflow_orchestrator:
            raise WorkflowError("Workflow orchestrator not initialized")
        
        # Execute workflow
        result = await workflow_orchestrator.smart_translate_workflow(
            text=params.text,
            target_language=params.target_language,
            quality_threshold=params.quality_threshold,
            terminology_names=params.terminology_names,
            auto_detect_language=params.auto_detect_language
        )
        
        return {
            "workflow_type": "smart_translation",
            "original_text": result.original_text,
            "translated_text": result.translated_text,
            "detected_language": result.detected_language,
            "target_language": result.target_language,
            "confidence_score": result.confidence_score,
            "quality_score": result.quality_score,
            "applied_terminologies": result.applied_terminologies,
            "language_pair_supported": result.language_pair_supported,
            "validation_issues": result.validation_issues,
            "suggestions": result.suggestions,
            "execution_time_ms": result.execution_time_ms,
            "workflow_steps": result.workflow_steps
        }
        
    except Exception as e:
        logger.error(f"Smart translate workflow failed: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def managed_batch_translation_workflow(params: ManagedBatchTranslationWorkflowParams) -> Dict[str, Any]:
    """
    Execute managed batch translation workflow with comprehensive monitoring and analytics.
    
    This workflow provides complete batch translation lifecycle management:
    1. Pre-validates language pairs and terminologies
    2. Starts batch translation job with S3 integration
    3. Monitors job progress with automated polling
    4. Collects performance metrics upon completion
    5. Returns comprehensive results with monitoring history
    
    Benefits:
    - Pre-validation of resources before job start
    - Automated monitoring with continuous progress tracking
    - Performance analytics and optimization insights
    - Comprehensive error handling and status reporting
    """
    try:
        if not workflow_orchestrator:
            raise WorkflowError("Workflow orchestrator not initialized")
        
        # Execute workflow
        result = await workflow_orchestrator.managed_batch_translation_workflow(
            input_s3_uri=params.input_s3_uri,
            output_s3_uri=params.output_s3_uri,
            data_access_role_arn=params.data_access_role_arn,
            job_name=params.job_name,
            source_language=params.source_language,
            target_languages=params.target_languages,
            terminology_names=params.terminology_names,
            content_type=params.content_type,
            monitor_interval=params.monitor_interval,
            max_monitoring_duration=params.max_monitoring_duration
        )
        
        return {
            "workflow_type": "managed_batch_translation",
            "job_id": result.job_id,
            "job_name": result.job_name,
            "status": result.status,
            "source_language": result.source_language,
            "target_languages": result.target_languages,
            "input_s3_uri": result.input_s3_uri,
            "output_s3_uri": result.output_s3_uri,
            "terminology_names": result.terminology_names,
            "pre_validation_results": result.pre_validation_results,
            "monitoring_history": result.monitoring_history,
            "performance_metrics": result.performance_metrics,
            "created_at": result.created_at.isoformat() if result.created_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "total_execution_time": result.total_execution_time,
            "workflow_steps": result.workflow_steps
        }
        
    except Exception as e:
        logger.error(f"Managed batch translation workflow failed: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def list_active_workflows() -> Dict[str, Any]:
    """
    List all currently active workflows.
    
    Returns information about workflows that are currently executing,
    including their current step and progress.
    """
    try:
        if not workflow_orchestrator:
            raise WorkflowError("Workflow orchestrator not initialized")
        
        active_workflows = workflow_orchestrator.list_active_workflows()
        
        return {
            "active_workflows": active_workflows,
            "total_count": len(active_workflows)
        }
        
    except Exception as e:
        logger.error(f"Failed to list active workflows: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get the current status of a specific workflow.
    
    Returns detailed information about workflow progress, current step,
    and any errors encountered.
    """
    try:
        if not workflow_orchestrator:
            raise WorkflowError("Workflow orchestrator not initialized")
        
        status = workflow_orchestrator.get_workflow_status(workflow_id)
        
        if status is None:
            return {
                "error": f"Workflow {workflow_id} not found",
                "error_type": "WorkflowNotFound"
            }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


def health_check() -> Dict[str, Any]:
    """
    Perform a health check of the server and its dependencies.
    
    Returns the health status of all components and AWS service connectivity.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check AWS client manager
        if aws_client_manager:
            try:
                aws_client_manager.validate_credentials()
                health_status["components"]["aws_client"] = "healthy"
            except Exception as e:
                health_status["components"]["aws_client"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
        else:
            health_status["components"]["aws_client"] = "not_initialized"
            health_status["status"] = "unhealthy"
        
        # Check translation service
        if translation_service:
            health_status["components"]["translation_service"] = "healthy"
        else:
            health_status["components"]["translation_service"] = "not_initialized"
            health_status["status"] = "unhealthy"
        
        # Check other services
        services = {
            "batch_manager": batch_manager,
            "terminology_manager": terminology_manager,
            "language_operations": language_operations,
            "secure_translation_service": secure_translation_service,
            "workflow_orchestrator": workflow_orchestrator
        }
        
        for service_name, service in services.items():
            if service:
                health_status["components"][service_name] = "healthy"
            else:
                health_status["components"][service_name] = "not_initialized"
                if health_status["status"] == "healthy":
                    health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def main() -> None:
    """Main entry point for the Amazon Translate MCP Server."""
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting Amazon Translate MCP Server...")
        
        # Load configuration
        config = load_config_from_env()
        logger.info(f"Configuration loaded: {config}")
        
        # Initialize services
        initialize_services()
        
        # Run the server
        logger.info("Amazon Translate MCP Server is ready to accept connections")
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()