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
import time
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


# Separate Batch Translation Tool Parameters

class TriggerBatchTranslationParams(BaseModel):
    """Parameters for trigger_batch_translation tool."""
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


class MonitorBatchTranslationParams(BaseModel):
    """Parameters for monitor_batch_translation tool."""
    job_id: str = Field(..., description="Translation job ID to monitor")
    output_s3_uri: str = Field(..., description="S3 URI for output location (for error analysis)")
    monitor_interval: int = Field(default=30, description="Monitoring interval in seconds")
    max_monitoring_duration: int = Field(default=3600, description="Maximum monitoring duration in seconds")


class AnalyzeBatchTranslationErrorsParams(BaseModel):
    """Parameters for analyze_batch_translation_errors tool."""
    job_id: str = Field(..., description="Failed translation job ID to analyze")
    output_s3_uri: str = Field(..., description="S3 URI for output location where error details are stored")


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
            "status": job_status.status,
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
                "status": job.status,
                "source_language": job.source_language_code,
                "target_languages": job.target_language_codes,
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
        
        # Convert terms to CSV format
        csv_content = "source,target\n"
        for term in params.terms:
            source = term.get("source", "")
            target = term.get("target", "")
            csv_content += f'"{source}","{target}"\n'
        
        # Create terminology data object
        terminology_data = TerminologyData(
            terminology_data=csv_content.encode('utf-8'),
            format="CSV"
        )
        
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
            "error_rate": metrics.error_rate
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


# Separate Batch Translation Tools

@mcp.tool()
async def trigger_batch_translation(params: TriggerBatchTranslationParams) -> Dict[str, Any]:
    """
    Trigger a batch translation job without monitoring.
    
    This tool starts a batch translation job and returns immediately with the job ID.
    Use this when you want to start a job and monitor it separately.
    
    Returns:
    - job_id: The unique identifier for the translation job
    - job_name: The name of the job
    - status: Initial job status (typically SUBMITTED)
    - validation_results: Pre-validation results for language pairs and terminologies
    """
    try:
        if not workflow_orchestrator:
            raise WorkflowError("Workflow orchestrator not initialized")
        
        if not batch_manager:
            raise BatchJobError("Batch manager not initialized")
        
        # Pre-validate language pairs
        loop = asyncio.get_event_loop()
        language_pairs = await loop.run_in_executor(
            None,
            language_operations.list_language_pairs
        )
        
        validation_results = {"supported_pairs": [], "unsupported_pairs": []}
        
        for target_lang in params.target_languages:
            pair_supported = any(
                pair.source_language == params.source_language and pair.target_language == target_lang
                for pair in language_pairs
            )
            
            if pair_supported:
                validation_results["supported_pairs"].append(f"{params.source_language}->{target_lang}")
            else:
                validation_results["unsupported_pairs"].append(f"{params.source_language}->{target_lang}")
        
        if validation_results["unsupported_pairs"]:
            raise ValidationError(
                f"Unsupported language pairs: {validation_results['unsupported_pairs']}",
                details=validation_results
            )
        
        # Validate terminologies if provided
        if params.terminology_names:
            terminologies_result = await loop.run_in_executor(
                None,
                terminology_manager.list_terminologies
            )
            available_terminologies = [t.name for t in terminologies_result.get('terminologies', [])]
            
            missing_terminologies = [
                name for name in params.terminology_names 
                if name not in available_terminologies
            ]
            
            if missing_terminologies:
                raise ValidationError(
                    f"Missing terminologies: {missing_terminologies}",
                    details={"missing": missing_terminologies, "available": available_terminologies}
                )
            
            validation_results["terminologies"] = {
                "requested": params.terminology_names,
                "available": available_terminologies,
                "validated": True
            }
        
        # Create configuration objects
        from .models import BatchInputConfig, BatchOutputConfig, JobConfig
        
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
        
        # Start the batch translation job
        job_id = await loop.run_in_executor(
            None,
            batch_manager.start_batch_translation,
            input_config,
            output_config,
            job_config
        )
        
        # Get initial job status
        job_status = await loop.run_in_executor(
            None,
            batch_manager.get_translation_job,
            job_id
        )
        
        return {
            "job_id": job_id,
            "job_name": params.job_name,
            "status": job_status.status,
            "source_language": params.source_language,
            "target_languages": params.target_languages,
            "input_s3_uri": params.input_s3_uri,
            "output_s3_uri": params.output_s3_uri,
            "terminology_names": params.terminology_names or [],
            "validation_results": validation_results,
            "created_at": job_status.created_at.isoformat() if job_status.created_at else None,
            "message": "Batch translation job started successfully. Use monitor_batch_translation to track progress."
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger batch translation: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def monitor_batch_translation(params: MonitorBatchTranslationParams) -> Dict[str, Any]:
    """
    Monitor a batch translation job until completion or failure.
    
    This tool continuously monitors a batch translation job and returns when it reaches
    a final state (COMPLETED, FAILED, or STOPPED). It provides progress updates and
    performs error analysis if the job fails.
    
    Returns:
    - job_id: The translation job ID
    - final_status: The final status of the job
    - monitoring_history: Complete history of status checks and progress
    - performance_metrics: Performance and timing metrics
    - error_analysis: Detailed error analysis if the job failed (includes S3 error details)
    """
    try:
        if not batch_manager:
            raise BatchJobError("Batch manager not initialized")
        
        if not workflow_orchestrator:
            raise WorkflowError("Workflow orchestrator not initialized")
        
        loop = asyncio.get_event_loop()
        monitoring_history = []
        start_time = time.time()
        
        logger.info(f"Starting monitoring for job {params.job_id}")
        
        # Monitor continuously until job reaches final state
        while True:
            current_time = time.time()
            
            # Get job status
            job_status = await loop.run_in_executor(
                None,
                batch_manager.get_translation_job,
                params.job_id
            )
            
            # Record monitoring data
            monitoring_entry = {
                "timestamp": datetime.now().isoformat(),
                "status": job_status.status,
                "progress": job_status.progress,
                "elapsed_time": current_time - start_time
            }
            monitoring_history.append(monitoring_entry)
            
            logger.info(
                f"Job {params.job_id} status: {job_status.status}, "
                f"progress: {job_status.progress}%, "
                f"elapsed: {current_time - start_time:.1f}s"
            )
            
            # Check if job reached final state
            if job_status.status in ["COMPLETED", "FAILED", "STOPPED"]:
                logger.info(f"Job {params.job_id} reached final state: {job_status.status}")
                break
            
            # Check if we've exceeded maximum monitoring duration
            if current_time - start_time > params.max_monitoring_duration:
                logger.warning(
                    f"Monitoring duration exceeded {params.max_monitoring_duration}s for job {params.job_id}. "
                    f"Job is still {job_status.status}. Stopping monitoring but job continues."
                )
                break
            
            # Wait before next check
            await asyncio.sleep(params.monitor_interval)
        
        # Analyze errors if job failed
        error_analysis = None
        if job_status and job_status.status == "FAILED":
            logger.info(f"Analyzing errors for failed job {params.job_id}")
            
            error_analysis = await workflow_orchestrator._analyze_job_errors(
                params.job_id, params.output_s3_uri, loop
            )
        
        # Calculate performance metrics
        total_monitoring_time = time.time() - start_time
        performance_metrics = {
            "total_monitoring_time": total_monitoring_time,
            "monitoring_checks": len(monitoring_history),
            "final_status": job_status.status if job_status else "UNKNOWN",
            "average_check_interval": total_monitoring_time / len(monitoring_history) if monitoring_history else 0
        }
        
        return {
            "job_id": params.job_id,
            "final_status": job_status.status if job_status else "UNKNOWN",
            "progress": job_status.progress if job_status else None,
            "monitoring_history": monitoring_history,
            "performance_metrics": performance_metrics,
            "error_analysis": error_analysis,
            "created_at": job_status.created_at.isoformat() if job_status and job_status.created_at else None,
            "completed_at": job_status.completed_at.isoformat() if job_status and job_status.completed_at else None,
            "message": f"Job monitoring completed. Final status: {job_status.status if job_status else 'UNKNOWN'}"
        }
        
    except Exception as e:
        logger.error(f"Failed to monitor batch translation: {e}")
        return {"error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def analyze_batch_translation_errors(params: AnalyzeBatchTranslationErrorsParams) -> Dict[str, Any]:
    """
    Analyze errors from a failed batch translation job.
    
    This tool examines the S3 output location for error detail files and provides
    comprehensive error analysis with actionable suggestions for resolution.
    
    Returns:
    - job_id: The failed job ID
    - error_files_found: List of error detail files discovered
    - error_details: Parsed error information from detail files
    - suggested_actions: Actionable suggestions to resolve the errors
    - error_summary: Summary of error patterns and root causes
    """
    try:
        if not workflow_orchestrator:
            raise WorkflowError("Workflow orchestrator not initialized")
        
        loop = asyncio.get_event_loop()
        
        logger.info(f"Analyzing errors for job {params.job_id}")
        
        # Perform error analysis
        error_analysis = await workflow_orchestrator._analyze_job_errors(
            params.job_id, params.output_s3_uri, loop
        )
        
        if not error_analysis:
            return {
                "job_id": params.job_id,
                "error": "No error details found for this job",
                "message": "Either the job didn't fail, or error details are not yet available in S3"
            }
        
        # Generate error summary
        error_summary = {
            "total_error_files": len(error_analysis.get("error_files_found", [])),
            "error_patterns": [],
            "affected_languages": [],
            "root_causes": []
        }
        
        # Analyze error patterns
        for detail in error_analysis.get("error_details", []):
            if "error_data" in detail:
                error_data = detail["error_data"]
                
                # Track affected languages
                source_lang = error_data.get("sourceLanguageCode")
                target_lang = error_data.get("targetLanguageCode")
                if source_lang and target_lang:
                    lang_pair = f"{source_lang}->{target_lang}"
                    if lang_pair not in error_summary["affected_languages"]:
                        error_summary["affected_languages"].append(lang_pair)
                
                # Analyze file-level errors
                if "details" in error_data:
                    for file_detail in error_data["details"]:
                        aux_data = file_detail.get("auxiliaryData", {})
                        if "error" in aux_data:
                            error_info = aux_data["error"]
                            error_code = error_info.get("errorCode", "Unknown")
                            error_message = error_info.get("errorMessage", "")
                            
                            # Categorize error patterns
                            if "utf-8" in error_message.lower():
                                if "UTF-8 Encoding Error" not in error_summary["error_patterns"]:
                                    error_summary["error_patterns"].append("UTF-8 Encoding Error")
                                    error_summary["root_causes"].append("Invalid file encoding or unsupported file format")
                            
                            if "format" in error_message.lower():
                                if "Unsupported Format" not in error_summary["error_patterns"]:
                                    error_summary["error_patterns"].append("Unsupported Format")
                                    error_summary["root_causes"].append("File format not supported by Amazon Translate")
        
        return {
            "job_id": params.job_id,
            "error_files_found": error_analysis.get("error_files_found", []),
            "error_details": error_analysis.get("error_details", []),
            "suggested_actions": error_analysis.get("suggested_actions", []),
            "error_summary": error_summary,
            "analysis_timestamp": datetime.now().isoformat(),
            "message": "Error analysis completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze batch translation errors: {e}")
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