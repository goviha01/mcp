# Amazon Translate MCP Server

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A Model Context Protocol (MCP) server that enables AI applications and tools to interact with Amazon Translate service using natural language for text translation, custom terminology management, and batch translation processing. The server provides a secure interface for managing translation jobs, custom terminologies, and language pairs while supporting 75+ languages and custom translation memory.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [MCP Client Setup](#mcp-client-setup)
- [Available Tools](#available-tools)
- [Usage Examples](#usage-examples)
- [AWS Permissions](#aws-permissions)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-language Support**: 75+ language pairs supported by Amazon Translate with validation
- **Real-time Translation**: Fast text translation with terminology support and confidence scoring
- **Batch Processing**: Efficient handling of large document collections with S3 integration
- **Custom Terminology**: Domain-specific translation consistency with CSV/TMX import support
- **Language Detection**: Automatic source language identification with alternative suggestions
- **Translation Validation**: Quality assessment with scoring and improvement suggestions
- **Security Features**: PII detection, content filtering, audit logging, and input validation
- **🆕 Intelligent Workflows**: Automated multi-step translation processes with orchestration
  - **Smart Translation Workflow**: Auto-detection, translation, and quality validation in one call
  - **Managed Batch Translation**: Complete batch job lifecycle with automated monitoring
- **Retry Logic**: Comprehensive error handling with exponential backoff and correlation tracking
- **Caching**: Configurable translation result and metadata caching for improved performance
- **File Security**: Secure file handling with size limits, extension validation, and temporary file cleanup
- **AWS Integration**: Native integration with S3, CloudWatch, IAM, and STS services
- **Health Monitoring**: Comprehensive health checks and service connectivity validation
- **MCP Compatible**: Works with Claude Desktop, Amazon Q Developer, and other MCP clients

## Installation

### Prerequisites

- Python 3.10 or higher
- AWS credentials configured
- An AWS account with Amazon Translate access

### Method 1: Using uvx (Recommended)

```bash
# Install and run directly
uvx awslabs.amazon-translate-mcp-server@latest

# Or install globally
uvx install awslabs.amazon-translate-mcp-server@latest
```

### Method 2: Using pip

```bash
# Install from PyPI
pip install awslabs.amazon-translate-mcp-server

# Run the server
python -m awslabs.amazon_translate_mcp_server.server
```

### Method 3: Using Docker

```bash
# Pull and run the Docker image
docker run -e AWS_REGION=us-east-1 \
  -v ~/.aws:/home/app/.aws:ro \
  awslabs/amazon-translate-mcp-server:latest
```

### Method 4: From Source

```bash
# Clone the repository
git clone https://github.com/awslabs/mcp.git
cd mcp/src/amazon-translate-mcp-server

# Install dependencies
pip install -e .

# Run the server (with configuration validation)
./start-server.sh

# Or run directly
python -m awslabs.amazon_translate_mcp_server.server
```

## Configuration

### Environment Variables

Set up your AWS credentials and configure the server:

```bash
# AWS Configuration (required)
export AWS_REGION=us-east-1
export AWS_PROFILE=your-profile

# Server Configuration (optional)
export FASTMCP_LOG_LEVEL=INFO
export TRANSLATE_CACHE_TTL=3600
export TRANSLATE_MAX_TEXT_LENGTH=10000
export TRANSLATE_BATCH_TIMEOUT=3600
export TRANSLATE_MAX_FILE_SIZE=10485760  # 10MB

# Feature Flags (optional)
export ENABLE_PII_DETECTION=false
export ENABLE_PROFANITY_FILTER=false
export ENABLE_CONTENT_FILTERING=false
export ENABLE_AUDIT_LOGGING=true
export ENABLE_TRANSLATION_CACHE=true

# File Handling (optional)
export TRANSLATE_ALLOWED_EXTENSIONS=".csv,.tmx,.txt"
export TRANSLATE_BLOCKED_PATTERNS=""
```

### Configuration Validation

The server performs comprehensive configuration validation at startup:

- AWS credentials and region validation
- Service connectivity testing
- Security configuration validation
- File handling limits verification
- Regex pattern validation for blocked content

Use the included `validate-config.py` script to test your configuration before starting the server.

See [ENVIRONMENT.md](ENVIRONMENT.md) for complete configuration options.

## MCP Client Setup

### Claude Desktop

Add to your Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "amazon-translate": {
      "command": "uvx",
      "args": ["awslabs.amazon-translate-mcp-server@latest"],
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_PROFILE": "default"
      }
    }
  }
}
```

### Other MCP Clients

See [MCP_CLIENT_CONFIG.md](MCP_CLIENT_CONFIG.md) for configuration examples for:
- VS Code with MCP extension
- Custom MCP implementations
- Development environments

## Documentation

- **[Deployment Guide](DEPLOYMENT.md)** - Complete deployment instructions for various environments
- **[Environment Variables](ENVIRONMENT.md)** - Configuration options and environment variables
- **[MCP Client Configuration](MCP_CLIENT_CONFIG.md)** - Setup examples for popular MCP clients
- **🆕 [Workflow Features](WORKFLOW_FEATURES.md)** - Detailed guide to intelligent workflow orchestration
- **[Changelog](CHANGELOG.md)** - Version history and changes

## Architecture Overview

The server follows a modular architecture with clear separation of concerns:

- **MCP Protocol Layer**: Handles MCP communication and tool registration
- **Service Layer**: Implements business logic for translation operations
- **AWS Integration Layer**: Manages AWS service interactions and authentication
- **Data Models**: Defines structured data types for requests and responses
- **Error Handling**: Provides comprehensive error management and logging

### Core Components

- **Translation Service**: Real-time text translation with terminology support and retry logic
- **Secure Translation Service**: Adds security features like PII detection and content filtering
- **Batch Manager**: Handles large-scale document translation jobs with S3 integration
- **Terminology Manager**: Manages custom terminology sets with CSV/TMX import support
- **Language Operations**: Provides language detection, metrics, and language pair validation
- **AWS Client Manager**: Manages AWS service clients with connection pooling and authentication

## Security and Compliance

- **Encryption**: All data encrypted in transit and at rest
- **Authentication**: Standard AWS credential chain support with validation
- **Authorization**: Fine-grained IAM permissions with role-based access
- **Audit Logging**: Comprehensive logging of all operations with correlation IDs
- **PII Detection**: Optional personally identifiable information detection and masking
- **Content Filtering**: Configurable profanity and content filtering with blocked patterns
- **Input Validation**: Comprehensive validation of all inputs with security checks
- **File Security**: Secure file handling with size limits and extension validation

## Performance and Scalability

- **Caching**: Intelligent caching of translation results and language pairs with configurable TTL
- **Concurrency**: Efficient handling of multiple simultaneous requests with async support
- **Rate Limiting**: Respects AWS service limits with exponential backoff and jitter
- **Retry Logic**: Comprehensive retry handling for transient failures with correlation tracking
- **Resource Management**: Proper cleanup and connection pooling with context managers
- **Monitoring**: Real-time performance metrics and health checks with CloudWatch integration

## Health Check

The server includes a comprehensive health check system that validates:

- AWS credential configuration and connectivity
- Amazon Translate service accessibility
- S3 service connectivity (for batch operations)
- All service component initialization status

The health check returns detailed status information for each component, helping diagnose configuration issues.

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure AWS credentials are properly configured
   - Check IAM permissions for required services
   - Verify AWS region configuration
   - Use the health check to validate connectivity

2. **Translation Failures**
   - Check text length limits (10,000 characters default, configurable)
   - Verify language pair support using `list_language_pairs`
   - Review terminology compatibility and conflicts
   - Check for blocked patterns in security configuration

3. **Batch Job Issues**
   - Ensure S3 bucket permissions are correct
   - Check input file formats and structure
   - Verify IAM role ARN format and permissions
   - Confirm data access role has proper S3 permissions

4. **Terminology Issues**
   - Validate CSV/TMX file format and encoding
   - Check language code compatibility
   - Ensure terminology names don't conflict
   - Verify file size limits (10MB default)

5. **Security and Validation Errors**
   - Review PII detection settings if enabled
   - Check content filtering and blocked patterns
   - Validate input text encoding and format
   - Ensure file extensions are allowed

For more troubleshooting information, see the [Deployment Guide](DEPLOYMENT.md).

## Development and Testing

### Running Tests

The server includes comprehensive test coverage:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=awslabs.amazon_translate_mcp_server

# Run specific test categories
pytest tests/test_translation_service.py
pytest tests/test_security.py
pytest tests/test_batch_manager.py
```

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/awslabs/mcp.git
cd mcp/src/amazon-translate-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run configuration validation
python validate-config.py

# Start server in development mode
python -m awslabs.amazon_translate_mcp_server.server
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on:

- Reporting bugs and feature requests
- Setting up the development environment
- Submitting pull requests
- Code style and testing requirements

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [AWS Labs MCP Documentation](https://awslabs.github.io/mcp/)
- **Issues**: [GitHub Issues](https://github.com/awslabs/mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/awslabs/mcp/discussions)

---

**Note**: This server requires an AWS account and appropriate permissions to use Amazon Translate services. AWS charges apply for translation operations.



## Available Tools

The server provides 16 MCP tools organized into five categories:

### Translation Operations
- **`translate_text`** - Real-time text translation with terminology support
  - Parameters: `text`, `source_language`, `target_language`, `terminology_names` (optional)
  - Returns: Translated text with metadata and confidence scores
- **`detect_language`** - Automatically identify source text language
  - Parameters: `text`
  - Returns: Detected language with confidence score and alternatives
- **`validate_translation`** - Perform quality checks on translations
  - Parameters: `original_text`, `translated_text`, `source_language`, `target_language`
  - Returns: Quality assessment and suggestions

### Batch Translation Operations
- **`start_batch_translation`** - Initialize batch translation jobs for large documents
  - Parameters: `input_s3_uri`, `output_s3_uri`, `data_access_role_arn`, `job_name`, `source_language`, `target_languages`, `content_type` (optional), `terminology_names` (optional)
  - Returns: Job ID for tracking progress
- **`get_translation_job`** - Monitor job status and retrieve results
  - Parameters: `job_id`
  - Returns: Job status, progress, and results location
- **`list_translation_jobs`** - View all translation tasks with filtering
  - Parameters: `status_filter` (optional), `max_results` (optional)
  - Returns: List of jobs with status and metadata

### Terminology Management
- **`list_terminologies`** - Browse available custom terminology sets
  - Returns: List of terminology sets with metadata
- **`create_terminology`** - Create new terminology for domain-specific translations
  - Parameters: `name`, `description`, `source_language`, `target_languages`, `terms`
  - Returns: Terminology creation status
- **`import_terminology`** - Import terminology from CSV or TMX files
  - Parameters: `name`, `description`, `file_content` (base64), `file_format`, `source_language`, `target_languages`
  - Returns: Import status and validation results
- **`get_terminology`** - Access detailed terminology information
  - Parameters: `name`
  - Returns: Terminology details, terms, and usage statistics

### Language Operations
- **`list_language_pairs`** - Show all 75+ supported language combinations
  - Returns: List of supported source-target language pairs
- **`get_language_metrics`** - View translation usage statistics
  - Parameters: `language_pair` (optional), `time_range` (optional)
  - Returns: Usage metrics and performance data

### 🆕 Intelligent Workflow Operations
- **`smart_translate_workflow`** - Automated translation with language detection and quality validation
  - Parameters: `text`, `target_language`, `quality_threshold` (optional), `terminology_names` (optional), `auto_detect_language` (optional)
  - Returns: Comprehensive translation results with detection, translation, and quality metrics
  - **Workflow Steps**: detect_language → validate_language_pair → translate_text → validate_translation
- **`managed_batch_translation_workflow`** - Complete batch translation lifecycle with automated monitoring
  - Parameters: `input_s3_uri`, `output_s3_uri`, `data_access_role_arn`, `job_name`, `source_language`, `target_languages`, `terminology_names` (optional), `content_type` (optional), `monitor_interval` (optional), `max_monitoring_duration` (optional)
  - Returns: Comprehensive job results with pre-validation, monitoring history, and performance analytics
  - **Workflow Steps**: validate_language_pairs → validate_terminologies → start_batch_job → monitor_job_progress → collect_metrics
- **`list_active_workflows`** - Monitor currently executing workflows
  - Returns: List of active workflows with current status and progress
- **`get_workflow_status`** - Get detailed status of a specific workflow
  - Parameters: `workflow_id`
  - Returns: Workflow progress, current step, and execution details

## Usage Examples

### Basic Text Translation

```python
# Using the MCP client
translate_text(
    text="Hello, how are you?",
    source_language="en",
    target_language="es"
)
# Returns: {
#   "translated_text": "Hola, ¿cómo estás?",
#   "source_language": "en",
#   "target_language": "es",
#   "applied_terminologies": [],
#   "confidence_score": 0.95
# }
```

### Language Detection

```python
# Detect language automatically
detect_language(text="Bonjour, comment allez-vous?")
# Returns: {
#   "detected_language": "fr",
#   "confidence_score": 0.99,
#   "alternative_languages": [("en", 0.01)]
# }
```

### Batch Translation

```python
# Start a batch translation job
start_batch_translation(
    input_s3_uri="s3://my-bucket/documents/",
    output_s3_uri="s3://my-bucket/translated/",
    data_access_role_arn="arn:aws:iam::123456789012:role/TranslateRole",
    job_name="marketing-materials-translation",
    source_language="en",
    target_languages=["es", "fr", "de"],
    content_type="text/plain"
)
# Returns: {
#   "job_id": "job-id-12345",
#   "status": "SUBMITTED",
#   "message": "Batch translation job started successfully"
# }

# Check job status
get_translation_job(job_id="job-id-12345")
# Returns: job status, progress, and results location
```

### Custom Terminology

```python
# Create custom terminology
create_terminology(
    name="medical-terms",
    description="Medical terminology for healthcare translations",
    source_language="en",
    target_languages=["es", "fr"],
    terms=[
        {"source": "hypertension", "target": "hipertensión"},
        {"source": "diabetes", "target": "diabetes"}
    ]
)

# Use terminology in translation
translate_text(
    text="The patient has hypertension and diabetes.",
    source_language="en",
    target_language="es",
    terminology_names=["medical-terms"]
)
```

### Translation Validation

```python
# Validate translation quality
validate_translation(
    original_text="Hello, world!",
    translated_text="Hola, mundo!",
    source_language="en",
    target_language="es"
)
# Returns: {
#   "is_valid": true,
#   "quality_score": 0.95,
#   "issues": [],
#   "suggestions": []
# }
```

### 🆕 Intelligent Workflow Examples

#### Smart Translation Workflow

```python
# Automated translation with language detection and quality validation
smart_translate_workflow(
    text="Bonjour, comment allez-vous? J'espère que vous passez une excellente journée!",
    target_language="en",
    quality_threshold=0.8,
    terminology_names=["business-terms"],  # Optional
    auto_detect_language=True
)
# Returns: {
#   "workflow_type": "smart_translation",
#   "original_text": "Bonjour, comment allez-vous?...",
#   "translated_text": "Hello, how are you? I hope you're having an excellent day!",
#   "detected_language": "fr",
#   "target_language": "en",
#   "confidence_score": 0.95,
#   "quality_score": 0.92,
#   "applied_terminologies": ["business-terms"],
#   "language_pair_supported": true,
#   "validation_issues": [],
#   "suggestions": [],
#   "execution_time_ms": 1250.5,
#   "workflow_steps": ["detect_language", "validate_language_pair", "translate_text", "validate_translation"]
# }
```

#### Managed Batch Translation Workflow

```python
# Complete batch translation lifecycle with automated monitoring
managed_batch_translation_workflow(
    input_s3_uri="s3://content-bucket/documents/",
    output_s3_uri="s3://output-bucket/translated/",
    data_access_role_arn="arn:aws:iam::123456789012:role/TranslateRole",
    job_name="website-localization-2024",
    source_language="en",
    target_languages=["es", "fr", "de", "it"],
    terminology_names=["ui-terms", "product-terms"],
    content_type="text/html",
    monitor_interval=30,  # Check every 30 seconds
    max_monitoring_duration=3600  # Monitor for up to 1 hour
)
# Returns: {
#   "workflow_type": "managed_batch_translation",
#   "job_id": "batch-translate-job-12345",
#   "job_name": "website-localization-2024",
#   "status": "COMPLETED",
#   "source_language": "en",
#   "target_languages": ["es", "fr", "de", "it"],
#   "pre_validation_results": {
#     "supported_pairs": ["en->es", "en->fr", "en->de", "en->it"],
#     "terminologies": {"validated": true}
#   },
#   "monitoring_history": [...],  # Detailed progress tracking
#   "performance_metrics": {...},  # Analytics and insights
#   "total_execution_time": 1200.5,
#   "workflow_steps": ["validate_language_pairs", "validate_terminologies", "start_batch_job", "monitor_job_progress", "collect_metrics"]
# }
```

#### Workflow Management

```python
# List currently active workflows
list_active_workflows()
# Returns: {
#   "active_workflows": [
#     {
#       "workflow_id": "smart_translate_1642234567890",
#       "workflow_type": "smart_translation",
#       "current_step": "translate_text",
#       "completed_steps": ["detect_language", "validate_language_pair"]
#     }
#   ],
#   "total_count": 1
# }

# Get specific workflow status
get_workflow_status(workflow_id="smart_translate_1642234567890")
# Returns detailed workflow progress and metadata
```

> **💡 Workflow Benefits**: The intelligent workflows eliminate the need for manual orchestration of multiple translation operations, provide automatic error handling and retry logic, include comprehensive monitoring and analytics, and ensure quality validation at each step.

## AWS Permissions

The server requires the following AWS IAM permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "translate:TranslateText",
                "translate:StartTextTranslationJob",
                "translate:DescribeTextTranslationJob",
                "translate:ListTextTranslationJobs",
                "translate:StopTextTranslationJob",
                "translate:CreateTerminology",
                "translate:GetTerminology",
                "translate:ListTerminologies",
                "translate:ImportTerminology",
translate:DeleteTerminology
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-translation-bucket/*",
                "arn:aws:s3:::your-translation-bucket"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "cloudwatch:GetMetricStatistics"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sts:GetCallerIdentity"
            ],
            "Resource": "*"
        }
    ]
}
```

For detailed permission setup, see [AWS IAM documentation](https://docs.aws.amazon.com/translate/latest/dg/security-iam.html).