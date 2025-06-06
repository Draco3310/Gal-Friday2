# Enhanced CLI Service Documentation

The Gal Friday CLI Service has been enhanced with comprehensive main guard functionality, providing robust command-line interface capabilities for production deployment and operational management.

## Overview

The enhanced CLI service replaces the previous empty `pass` statement with a full-featured command-line interface that includes:

- **Comprehensive argument parsing** with multiple options and modes
- **Configuration validation** and health checks
- **Graceful startup and shutdown** with signal handling
- **Multiple execution modes** (production, example, daemon)
- **Professional logging** with configurable levels and file output
- **Error handling and recovery** for operational resilience

## Quick Start

### Basic Usage

```bash
# Start CLI service with default settings
python -m gal_friday.cli_service

# Run example/demo mode
python -m gal_friday.cli_service --example

# Show help
python -m gal_friday.cli_service --help

# Check version
python -m gal_friday.cli_service --version
```

### Common Operations

```bash
# Validate configuration
python -m gal_friday.cli_service --validate-config --config config/production.yaml

# Perform health check
python -m gal_friday.cli_service --health-check

# Start with custom logging
python -m gal_friday.cli_service --log-level DEBUG --log-file logs/cli.log

# Run in daemon mode (requires python-daemon package)
python -m gal_friday.cli_service --daemon --config config/production.yaml
```

## Command Line Options

### Configuration Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to configuration file | `config/default.yaml` |
| `--log-level` | `-l` | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` |
| `--log-file` | | Path to log file (optional, logs to console by default) | None |

### Service Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--port` | `-p` | CLI service port | `8080` |
| `--host` | | CLI service host | `localhost` |
| `--mode` | `-m` | Trading mode (live_trading, paper_trading, backtesting, data_collection) | None |

### Operational Commands

| Option | Description |
|--------|-------------|
| `--health-check` | Perform comprehensive health check and exit |
| `--validate-config` | Validate configuration file and exit |
| `--example` | Run example/demo mode with mock services |
| `--daemon` | Run as daemon process |
| `--version` | Show version information |

## Execution Modes

### 1. Production Mode (Default)

```bash
python -m gal_friday.cli_service --config config/production.yaml
```

**Features:**
- Full service initialization
- Real configuration loading
- Production logging
- Signal handling for graceful shutdown
- Health monitoring

### 2. Example/Demo Mode

```bash
python -m gal_friday.cli_service --example
```

**Features:**
- Uses mock services for demonstration
- Safe for testing and development
- Shows CLI functionality without requiring real infrastructure
- Automatically shuts down after demonstration

### 3. Daemon Mode

```bash
python -m gal_friday.cli_service --daemon --config config/production.yaml
```

**Features:**
- Runs as background daemon process
- PID file management
- Proper process detachment
- Requires `python-daemon` package

### 4. Validation Mode

```bash
python -m gal_friday.cli_service --validate-config --config config/test.yaml
```

**Features:**
- Validates configuration without starting services
- Quick feedback on configuration issues
- Safe for CI/CD pipelines

### 5. Health Check Mode

```bash
python -m gal_friday.cli_service --health-check
```

**Features:**
- Comprehensive system health validation
- Configuration check
- Directory structure verification
- Database connectivity test (if configured)

## Configuration

### Configuration File Structure

The CLI service expects a YAML configuration file with the following structure:

```yaml
database:
  host: localhost
  port: 5432
  username: gal_friday
  database: trading_system

logging:
  level: INFO
  file: logs/application.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

services:
  cli_port: 8080
  cli_host: localhost
  
trading:
  mode: paper_trading
  initial_capital: 100000
```

### Environment Variables

The CLI service supports the following environment variables:

- `GAL_FRIDAY_CONFIG`: Default configuration file path
- `GAL_FRIDAY_LOG_LEVEL`: Default logging level
- `GAL_FRIDAY_MODE`: Default trading mode

## Available CLI Commands

Once the CLI service is running, you can use these interactive commands:

### Core Commands

| Command | Description |
|---------|-------------|
| `status` | Show system status and portfolio information |
| `halt [reason]` | Halt trading with optional reason |
| `resume` | Resume trading from halt state |
| `stop` | Gracefully shutdown the application |

### Recovery Commands

| Command | Description |
|---------|-------------|
| `recovery_status` | Show recovery items and status |
| `complete_recovery_item <id> <user>` | Mark recovery item as complete |

### Help and Information

| Command | Description |
|---------|-------------|
| `--help` | Show available commands and options |

## Signal Handling

The enhanced CLI service provides robust signal handling:

| Signal | Action |
|--------|--------|
| `SIGINT` (Ctrl+C) | Graceful shutdown |
| `SIGTERM` | Graceful shutdown |
| `SIGHUP` (Unix) | Graceful shutdown |

## Logging

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational information
- **WARNING**: Warning messages for attention
- **ERROR**: Error conditions that don't stop operation
- **CRITICAL**: Critical errors that may cause shutdown

### Log Format

```
2024-01-15 10:30:45 - gal_friday.cli_service - INFO - CLI service started on localhost:8080
2024-01-15 10:30:46 - gal_friday.cli_service - DEBUG - Processing command: status
```

### File Logging

```bash
# Log to file
python -m gal_friday.cli_service --log-file logs/cli_service.log

# Log with rotation (requires additional configuration)
python -m gal_friday.cli_service --log-file logs/cli_service.log --log-level DEBUG
```

## Health Checks

The health check functionality validates:

1. **Configuration File**: Existence and basic structure
2. **Required Directories**: Creation of logs, data, config directories
3. **Database Configuration**: Presence of database settings
4. **Service Dependencies**: Availability of required services

### Health Check Output

```bash
$ python -m gal_friday.cli_service --health-check
🏥 Performing health check...
✅ Health check passed
```

## Error Handling

The enhanced CLI service provides comprehensive error handling:

### Configuration Errors

```bash
$ python -m gal_friday.cli_service --config missing.yaml
❌ Configuration validation failed
```

### Service Errors

- Graceful degradation for missing services
- Comprehensive error logging
- Automatic recovery attempts
- Clean shutdown on critical errors

## Development and Testing

### Running Tests

```bash
# Run example to test functionality
python examples/cli_service_example.py

# Test specific features
python -m gal_friday.cli_service --validate-config --config test_config.yaml
python -m gal_friday.cli_service --health-check
```

### Mock Services

The CLI service includes comprehensive mock services for development:

- `MockLoggerService`: Logging simulation
- `MockMonitoringService`: System monitoring simulation
- `MockPortfolioManager`: Portfolio management simulation
- `MockConfigManager`: Configuration management simulation

## Integration with Existing System

### Service Integration

The enhanced CLI service integrates with existing system components:

```python
from gal_friday.cli_service import CLIService, CLIServiceRunner

# Create production services
monitoring_service = MonitoringService(...)
portfolio_manager = PortfolioManager(...)
logger_service = LoggerService(...)

# Initialize CLI service
cli_service = CLIService(
    monitoring_service=monitoring_service,
    main_app_controller=app_controller,
    logger_service=logger_service,
    portfolio_manager=portfolio_manager
)

# Start CLI service
await cli_service.start()
```

### Configuration Integration

```python
from gal_friday.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager("config/production.yaml")

# Use with CLI service runner
runner = CLIServiceRunner()
# Configuration is automatically loaded from command line arguments
```

## Deployment

### Production Deployment

```bash
# Basic production start
python -m gal_friday.cli_service --config config/production.yaml --log-level INFO

# Daemon mode for server deployment
python -m gal_friday.cli_service --daemon --config config/production.yaml --log-file /var/log/gal_friday/cli.log
```

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "-m", "gal_friday.cli_service", "--config", "config/docker.yaml"]
```

### Systemd Service

```ini
# /etc/systemd/system/gal-friday-cli.service
[Unit]
Description=Gal Friday CLI Service
After=network.target

[Service]
Type=simple
User=galfridy
WorkingDirectory=/opt/gal-friday
ExecStart=/opt/gal-friday/venv/bin/python -m gal_friday.cli_service --config config/production.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'daemon'`
**Solution**: Install python-daemon: `pip install python-daemon`

**Issue**: Permission denied for PID file
**Solution**: Ensure write permissions for `/var/run/` or use custom PID file location

**Issue**: Configuration validation fails
**Solution**: Check YAML syntax and required sections in configuration file

### Debug Mode

```bash
# Enable debug logging for troubleshooting
python -m gal_friday.cli_service --log-level DEBUG --example
```

### Logs Analysis

```bash
# Check logs for errors
tail -f logs/cli_service.log | grep ERROR

# Monitor startup
python -m gal_friday.cli_service --log-level DEBUG 2>&1 | tee startup.log
```

## Best Practices

### Configuration Management

- Use version control for configuration files
- Validate configurations in CI/CD pipelines
- Use environment-specific configuration files
- Implement configuration templating for deployments

### Logging

- Use appropriate log levels for different environments
- Implement log rotation for production deployments
- Monitor logs for operational insights
- Include contextual information in log messages

### Security

- Restrict access to configuration files
- Use secure file permissions for log files
- Implement proper authentication for production deployments
- Regularly update dependencies

### Monitoring

- Implement health check endpoints
- Monitor CLI service availability
- Set up alerting for service failures
- Track performance metrics

## Migration from Legacy CLI

### From Empty Pass Statement

The original CLI service had an empty `pass` statement in the main guard:

```python
# Old implementation
if __name__ == "__main__":
    pass
```

The enhanced version provides full functionality:

```python
# New implementation
if __name__ == "__main__":
    main()
```

### Backward Compatibility

The enhanced CLI service maintains backward compatibility:

- Existing `example_main()` function still available via `--example` flag
- All original CLI commands continue to work
- Mock services preserved for testing
- Original service initialization patterns supported

## API Reference

### CLIServiceRunner

Main class for CLI service lifecycle management.

#### Methods

- `setup_argument_parser()`: Configure command line arguments
- `setup_logging(level, file)`: Configure logging
- `setup_signal_handlers()`: Configure signal handling
- `validate_configuration(path)`: Validate configuration file
- `perform_health_check(path)`: Run health checks
- `start_cli_service(args)`: Start CLI service
- `shutdown_cli_service()`: Graceful shutdown

### main()

Main entry point function that replaces the empty pass statement.

**Features:**
- Argument parsing
- Logging setup
- Signal handling
- Error handling
- Service lifecycle management

For complete API documentation, see the source code docstrings and type hints. 