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

# Multi-stage build for Amazon Translate MCP Server
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv-requirements.txt ./

# Install dependencies
RUN uv pip install --system -r uv-requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID for security
RUN groupadd -r app --gid=1000 && \
    useradd -r -g app --uid=1000 --home-dir=/app --shell=/bin/bash app

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY awslabs/ ./awslabs/
COPY docker-healthcheck.sh ./
COPY validate-config.py ./
COPY start-server.sh ./

# Make scripts executable
RUN chmod +x docker-healthcheck.sh validate-config.py start-server.sh

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables with defaults
ENV PYTHONPATH=/app
ENV FASTMCP_LOG_LEVEL=INFO
ENV TRANSLATE_MAX_TEXT_LENGTH=10000
ENV TRANSLATE_BATCH_TIMEOUT=3600
ENV TRANSLATE_CACHE_TTL=3600
ENV ENABLE_TRANSLATION_CACHE=true
ENV ENABLE_AUDIT_LOGGING=true
ENV ENABLE_PII_DETECTION=false
ENV ENABLE_PROFANITY_FILTER=false
ENV ENABLE_CONTENT_FILTERING=false

# Health check with improved configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ./docker-healthcheck.sh

# Expose port (if needed for HTTP mode)
EXPOSE 8000

# Add labels for better container management
LABEL maintainer="AWSLabs MCP <203918161+awslabs-mcp@users.noreply.github.com>"
LABEL version="1.0.0"
LABEL description="Amazon Translate MCP Server"
LABEL org.opencontainers.image.source="https://github.com/awslabs/mcp"
LABEL org.opencontainers.image.documentation="https://awslabs.github.io/mcp/servers/amazon-translate-mcp-server/"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Default command with configuration validation
CMD ["./start-server.sh"]