#!/usr/bin/env python3
"""
Example usage of Amazon Translate MCP Server Workflow Features.

This file demonstrates how to use the Smart Translation Workflow and
Managed Batch Translation Workflow features.
"""

import asyncio
import json
from typing import Dict, Any


async def example_smart_translate_workflow():
    """
    Example of using the Smart Translation Workflow.
    
    This workflow automatically:
    1. Detects the source language
    2. Validates language pair support
    3. Translates the text
    4. Validates translation quality
    """
    
    # Example parameters for smart translation workflow
    workflow_params = {
        "text": "Bonjour, comment allez-vous? J'espère que vous passez une excellente journée!",
        "target_language": "en",
        "quality_threshold": 0.8,
        "terminology_names": [],  # Optional custom terminology
        "auto_detect_language": True
    }
    
    print("=== Smart Translation Workflow Example ===")
    print(f"Input text: {workflow_params['text']}")
    print(f"Target language: {workflow_params['target_language']}")
    print(f"Quality threshold: {workflow_params['quality_threshold']}")
    print()
    
    # In a real implementation, you would call the MCP tool:
    # result = await mcp_client.call_tool("smart_translate_workflow", workflow_params)
    
    # Example expected result structure:
    expected_result = {
        "workflow_type": "smart_translation",
        "original_text": workflow_params["text"],
        "translated_text": "Hello, how are you? I hope you're having an excellent day!",
        "detected_language": "fr",
        "target_language": "en",
        "confidence_score": 0.95,
        "quality_score": 0.92,
        "applied_terminologies": [],
        "language_pair_supported": True,
        "validation_issues": [],
        "suggestions": [],
        "execution_time_ms": 1250.5,
        "workflow_steps": [
            "detect_language",
            "validate_language_pair", 
            "translate_text",
            "validate_translation"
        ]
    }
    
    print("Expected workflow result:")
    print(json.dumps(expected_result, indent=2))
    print()
    
    return expected_result


async def example_managed_batch_translation_workflow():
    """
    Example of using the Managed Batch Translation Workflow.
    
    This workflow automatically:
    1. Pre-validates language pairs and terminologies
    2. Starts the batch translation job
    3. Monitors job progress with automated polling
    4. Collects performance metrics upon completion
    """
    
    # Example parameters for managed batch translation workflow
    workflow_params = {
        "input_s3_uri": "s3://my-content-bucket/documents/",
        "output_s3_uri": "s3://my-output-bucket/translated/",
        "data_access_role_arn": "arn:aws:iam::123456789012:role/TranslateServiceRole",
        "job_name": "website-localization-2024",
        "source_language": "en",
        "target_languages": ["es", "fr", "de", "it"],
        "terminology_names": ["ui-terms", "product-terms"],
        "content_type": "text/html",
        "monitor_interval": 30,
        "max_monitoring_duration": 3600
    }
    
    print("=== Managed Batch Translation Workflow Example ===")
    print(f"Input S3 URI: {workflow_params['input_s3_uri']}")
    print(f"Output S3 URI: {workflow_params['output_s3_uri']}")
    print(f"Source language: {workflow_params['source_language']}")
    print(f"Target languages: {', '.join(workflow_params['target_languages'])}")
    print(f"Terminologies: {', '.join(workflow_params['terminology_names'])}")
    print(f"Monitor interval: {workflow_params['monitor_interval']} seconds")
    print()
    
    # In a real implementation, you would call the MCP tool:
    # result = await mcp_client.call_tool("managed_batch_translation_workflow", workflow_params)
    
    # Example expected result structure:
    expected_result = {
        "workflow_type": "managed_batch_translation",
        "job_id": "batch-translate-job-12345",
        "job_name": workflow_params["job_name"],
        "status": "COMPLETED",
        "source_language": "en",
        "target_languages": ["es", "fr", "de", "it"],
        "input_s3_uri": workflow_params["input_s3_uri"],
        "output_s3_uri": workflow_params["output_s3_uri"],
        "terminology_names": workflow_params["terminology_names"],
        "pre_validation_results": {
            "supported_pairs": ["en->es", "en->fr", "en->de", "en->it"],
            "unsupported_pairs": [],
            "terminologies": {
                "requested": ["ui-terms", "product-terms"],
                "available": ["ui-terms", "product-terms", "legal-terms"],
                "validated": True
            }
        },
        "monitoring_history": [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "status": "SUBMITTED",
                "progress": 0,
                "elapsed_time": 0
            },
            {
                "timestamp": "2024-01-15T10:05:00Z", 
                "status": "IN_PROGRESS",
                "progress": 25,
                "elapsed_time": 300
            },
            {
                "timestamp": "2024-01-15T10:15:00Z",
                "status": "IN_PROGRESS", 
                "progress": 75,
                "elapsed_time": 900
            },
            {
                "timestamp": "2024-01-15T10:20:00Z",
                "status": "COMPLETED",
                "progress": 100,
                "elapsed_time": 1200
            }
        ],
        "performance_metrics": {
            "language_pairs": {
                "en-es": {
                    "translation_count": 1250,
                    "character_count": 125000,
                    "average_response_time": 0.85,
                    "error_rate": 0.001
                },
                "en-fr": {
                    "translation_count": 1250,
                    "character_count": 130000,
                    "average_response_time": 0.92,
                    "error_rate": 0.002
                }
            },
            "total_monitoring_time": 1200,
            "monitoring_checks": 4,
            "final_status": "COMPLETED"
        },
        "created_at": "2024-01-15T10:00:00Z",
        "completed_at": "2024-01-15T10:20:00Z",
        "total_execution_time": 1200.5,
        "workflow_steps": [
            "validate_language_pairs",
            "validate_terminologies",
            "start_batch_job",
            "monitor_job_progress",
            "collect_metrics"
        ]
    }
    
    print("Expected workflow result:")
    print(json.dumps(expected_result, indent=2))
    print()
    
    return expected_result


async def example_workflow_management():
    """
    Example of using workflow management tools.
    """
    
    print("=== Workflow Management Examples ===")
    
    # List active workflows
    print("1. Listing active workflows:")
    active_workflows_result = {
        "active_workflows": [
            {
                "workflow_id": "smart_translate_1642234567890",
                "workflow_type": "smart_translation",
                "started_at": "2024-01-15T10:30:00Z",
                "current_step": "translate_text",
                "completed_steps": ["detect_language", "validate_language_pair"],
                "error_count": 0,
                "retry_count": 0,
                "metadata": {
                    "text_length": 65,
                    "target_language": "en",
                    "quality_threshold": 0.8
                }
            }
        ],
        "total_count": 1
    }
    print(json.dumps(active_workflows_result, indent=2))
    print()
    
    # Get specific workflow status
    print("2. Getting specific workflow status:")
    workflow_status_result = {
        "workflow_id": "smart_translate_1642234567890",
        "workflow_type": "smart_translation",
        "started_at": "2024-01-15T10:30:00Z",
        "current_step": "validate_translation",
        "completed_steps": ["detect_language", "validate_language_pair", "translate_text"],
        "error_count": 0,
        "retry_count": 0,
        "metadata": {
            "text_length": 65,
            "target_language": "en",
            "quality_threshold": 0.8
        }
    }
    print(json.dumps(workflow_status_result, indent=2))
    print()


async def main():
    """Run all workflow examples."""
    print("Amazon Translate MCP Server - Workflow Examples")
    print("=" * 50)
    print()
    
    await example_smart_translate_workflow()
    print()
    
    await example_managed_batch_translation_workflow()
    print()
    
    await example_workflow_management()
    
    print("=" * 50)
    print("Examples completed!")
    print()
    print("To use these workflows in practice:")
    print("1. Start the Amazon Translate MCP Server")
    print("2. Connect your MCP client (e.g., VS Code, Amazon Q Developer)")
    print("3. Call the workflow tools with the parameters shown above")
    print("4. Monitor progress using the workflow management tools")


if __name__ == "__main__":
    asyncio.run(main())