#!/usr/bin/env python3
"""
Configuration validation script for Amazon Translate MCP Server.

This script can be used to validate configuration before starting the server
or as part of deployment health checks.
"""

import sys
import os
import logging
from typing import Optional

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from awslabs.amazon_translate_mcp_server.config import (
        validate_startup_configuration,
        print_configuration_summary
    )
except ImportError as e:
    print(f"Error importing configuration module: {e}")
    print("Make sure the package is properly installed.")
    sys.exit(1)


def main():
    """Main validation function."""
    
    print("Amazon Translate MCP Server - Configuration Validation")
    print("=" * 60)
    
    try:
        # Validate configuration
        config = validate_startup_configuration()
        
        # Print configuration summary
        print_configuration_summary(config)
        
        print("✅ Configuration validation completed successfully!")
        print("\nThe server is ready to start with the current configuration.")
        
        return 0
        
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        print("\nPlease check your environment variables and try again.")
        return 1
        
    except RuntimeError as e:
        print(f"❌ Runtime validation failed: {e}")
        print("\nPlease check your AWS credentials and permissions.")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}")
        print("\nPlease check the server logs for more details.")
        return 1


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    sys.exit(main())