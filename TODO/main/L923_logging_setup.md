# Task: Implement logging setup and command line interface initialization.

### 1. Context
- **File:** `main.py`
- **Line:** `923`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing logging setup and command line interface initialization.

### 2. Problem Statement
Without proper logging setup and command line interface initialization, the application lacks essential debugging capabilities, operational visibility, and user-friendly interaction mechanisms. This prevents effective troubleshooting, monitoring, and flexible configuration during application startup.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Logging Configuration Framework:** Comprehensive logging setup with multiple handlers and formatters
2. **Build CLI Argument Parser:** Robust command line interface with validation and help system
3. **Implement Dynamic Log Level Control:** Runtime log level adjustment based on CLI arguments and configuration
4. **Add Structured Logging:** JSON and structured logging for enterprise monitoring systems
5. **Create Log Rotation and Archival:** Automatic log rotation with configurable retention policies
6. **Build Configuration Override System:** CLI arguments that override configuration file settings

#### b. Pseudocode or Implementation Sketch
```python
import argparse
import logging
import logging.config
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os

class LoggingSetup:
    """Enterprise-grade logging configuration and initialization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None
        
    def setup_logging(self, log_level: str = None, log_file: str = None, 
                     enable_json: bool = False, enable_console: bool = True) -> logging.Logger:
        """
        Setup comprehensive logging system
        Replace TODO with enterprise logging configuration
        """
        
        # Determine log level
        if log_level:
            level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO').upper(), logging.INFO)
        
        # Create formatters
        formatters = self._create_formatters(enable_json)
        
        # Create handlers
        handlers = self._create_handlers(log_file, enable_console, formatters)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True  # Override any existing configuration
        )
        
        # Configure specific loggers
        self._configure_library_loggers()
        
        # Create application logger
        logger = logging.getLogger('gal_friday')
        logger.setLevel(level)
        
        # Log initial startup information
        logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}")
        logger.info(f"Log handlers: {[type(h).__name__ for h in handlers]}")
        
        self.logger = logger
        return logger

class CLIParser:
    """Command line interface parser with comprehensive options"""
    
    def __init__(self):
        self.parser = None
        self._setup_parser()
    
    def _setup_parser(self) -> None:
        """Setup command line argument parser"""
        
        self.parser = argparse.ArgumentParser(
            description='Gal Friday - Enterprise Trading System',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Operational mode arguments
        mode_group = self.parser.add_argument_group('Operational Mode')
        mode_group.add_argument(
            '--mode', '-m',
            choices=['live', 'paper', 'backtest', 'data-collection', 'monitoring'],
            default='paper',
            help='Trading mode (default: paper)'
        )
        
        # Logging arguments
        log_group = self.parser.add_argument_group('Logging')
        log_group.add_argument(
            '--log-level', '-l',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Set logging level'
        )
        log_group.add_argument(
            '--log-file',
            type=str,
            help='Path to log file'
        )
        log_group.add_argument(
            '--enable-json-logs',
            action='store_true',
            help='Enable JSON structured logging'
        )
        
        # Configuration arguments
        config_group = self.parser.add_argument_group('Configuration')
        config_group.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file'
        )
    
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse command line arguments with validation"""
        
        parsed_args = self.parser.parse_args(args)
        
        # Validate arguments
        self._validate_args(parsed_args)
        
        return parsed_args

def initialize_application(config: Dict[str, Any]) -> tuple:
    """
    Initialize application with logging and CLI setup
    Main entry point replacing TODO
    """
    
    # Setup command line interface
    cli_parser = CLIParser()
    args = cli_parser.parse_args()
    
    # Setup logging
    logging_setup = LoggingSetup(config)
    logger = logging_setup.setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        enable_json=args.enable_json_logs
    )
    
    # Log startup information
    logger.info("Gal Friday Trading System Starting")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config or 'default'}")
    
    return logger, args
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of logging setup failures; fallback configurations when log files inaccessible; CLI validation with helpful error messages
- **Configuration:** Flexible logging configuration with environment-specific settings; CLI argument precedence over config files; validation of all input parameters
- **Testing:** Unit tests for CLI argument parsing; integration tests for logging output; validation of log rotation and archival
- **Dependencies:** Standard library logging and argparse modules; configuration management system; file system access for log files

### 4. Acceptance Criteria
- [ ] Logging system supports multiple output destinations (console, file, JSON, error logs) with appropriate formatters
- [ ] Log rotation automatically manages file sizes and retention policies to prevent disk space issues
- [ ] CLI interface provides comprehensive argument parsing with validation and helpful error messages
- [ ] Dynamic log level control allows runtime adjustment of logging verbosity
- [ ] Structured JSON logging enables integration with enterprise monitoring and log aggregation systems
- [ ] Configuration override system allows CLI arguments to supersede configuration file settings
- [ ] Performance testing shows minimal logging overhead in production environments
- [ ] Help system provides clear documentation and usage examples for all CLI options
- [ ] TODO placeholder is completely replaced with production-ready implementation
