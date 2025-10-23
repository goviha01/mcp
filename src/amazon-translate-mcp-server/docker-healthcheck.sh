#!/bin/bash

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

# Health check script for Amazon Translate MCP Server Docker container

set -e

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if the server process is running
log "Checking if Amazon Translate MCP Server process is running..."
if ! pgrep -f "awslabs.amazon_translate_mcp_server.server" > /dev/null; then
    echo -e "${RED}ERROR: Amazon Translate MCP Server process not found${NC}"
    exit 1
fi
log "✓ Server process is running"

# Check Python environment
log "Checking Python environment..."
if ! command_exists python3; then
    echo -e "${RED}ERROR: Python3 not found${NC}"
    exit 1
fi

# Check if required modules can be imported
log "Checking required Python modules..."
if ! python3 -c "import boto3, fastmcp, pydantic" 2>/dev/null; then
    echo -e "${RED}ERROR: Required Python modules not available${NC}"
    exit 1
fi
log "✓ Required Python modules available"

# Check configuration loading
log "Checking configuration loading..."
if ! python3 -c "from awslabs.amazon_translate_mcp_server.config import get_config; get_config()" 2>/dev/null; then
    echo -e "${RED}ERROR: Configuration loading failed${NC}"
    exit 1
fi
log "✓ Configuration loaded successfully"

# Check AWS credentials availability (non-blocking)
log "Checking AWS credentials..."
if python3 -c "
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
try:
    client = boto3.client('translate')
    # Try to list supported languages as a lightweight check
    client.list_languages()
    print('AWS credentials are valid')
except (NoCredentialsError, PartialCredentialsError):
    print('WARNING: AWS credentials not configured')
    exit(2)  # Warning exit code
except Exception as e:
    print(f'WARNING: AWS credential check failed: {e}')
    exit(2)  # Warning exit code
" 2>/dev/null; then
    log "✓ AWS credentials are valid"
elif [ $? -eq 2 ]; then
    echo -e "${YELLOW}WARNING: AWS credentials may not be properly configured${NC}"
    log "This might be expected in some environments (IAM roles, etc.)"
else
    echo -e "${RED}ERROR: AWS credential check failed${NC}"
    exit 1
fi

# Check memory usage (optional)
if command_exists free; then
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    log "Memory usage: ${MEMORY_USAGE}%"
    
    # Alert if memory usage is very high
    if (( $(echo "$MEMORY_USAGE > 90.0" | bc -l) )); then
        echo -e "${YELLOW}WARNING: High memory usage detected (${MEMORY_USAGE}%)${NC}"
    fi
fi

# Check disk space (optional)
if command_exists df; then
    DISK_USAGE=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    log "Disk usage: ${DISK_USAGE}%"
    
    # Alert if disk usage is very high
    if [ "$DISK_USAGE" -gt 90 ]; then
        echo -e "${YELLOW}WARNING: High disk usage detected (${DISK_USAGE}%)${NC}"
    fi
fi

echo -e "${GREEN}Health check passed: Amazon Translate MCP Server is healthy${NC}"
exit 0