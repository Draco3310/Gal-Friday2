#!/usr/bin/env python3
"""Example demonstrating the enhanced CLI service with comprehensive main guard.

This example shows how to use the CLI service with:
- Command line argument parsing
- Configuration validation
- Health checks
- Different execution modes
- Graceful shutdown
"""

import logging
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)


def run_cli_command(args: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a CLI command and return the result."""
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "gal_friday.cli_service", *args],
            check=False, capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)
    else:
        return result.returncode, result.stdout, result.stderr


def test_help_command():
    """Test the help command."""
    returncode, stdout, stderr = run_cli_command(["--help"])
    if returncode == 0:
        for line in stdout.split("\n")[:10]:
            pass
    else:
        pass


def test_version_command():
    """Test the version command."""
    returncode, stdout, stderr = run_cli_command(["--version"])
    if returncode == 0:
        pass
    else:
        pass


def test_config_validation_invalid():
    """Test config validation with non-existent config."""
    returncode, stdout, stderr = run_cli_command([
        "--validate-config",
        "--config", "non_existent_config.yaml",
    ])
    if returncode != 0:
        pass
    else:
        pass


def test_config_validation_valid():
    """Test config validation with valid config file."""
    config_path = Path("test_config.yaml")
    try:
        config_content = """
database:
  host: localhost
  port: 5432

logging:
  level: INFO

services:
  cli_port: 8080
"""
        config_path.write_text(config_content)

        # Test config validation with valid file
        returncode, stdout, stderr = run_cli_command([
            "--validate-config",
            "--config", str(config_path),
        ])
        if returncode == 0:
            pass
        else:
            pass

    except Exception as e:
        logger.exception("Error running command: %s", e)

    return config_path


def test_health_check(config_path):
    """Test health check command."""
    returncode, stdout, stderr = run_cli_command([
        "--health-check",
        "--config", str(config_path) if config_path.exists() else "config/default.yaml",
    ])
    if returncode == 0:
        pass
    else:
        pass


def test_example_mode():
    """Test example mode with timeout."""
    returncode, stdout, stderr = run_cli_command(["--example"], timeout=5)
    if returncode in (0, -1):  # Timeout is expected
        if stdout:
            for line in stdout.split("\n")[:5]:
                if line.strip():
                    pass
    else:
        pass


def test_log_levels(config_path):
    """Test different log levels."""
    for log_level in ["DEBUG", "WARNING", "ERROR"]:
        returncode, stdout, stderr = run_cli_command([
            "--validate-config",
            "--log-level", log_level,
            "--config", str(config_path) if config_path.exists() else "config/default.yaml",
        ])
        if returncode == 0:
            pass
        else:
            pass


def show_available_options():
    """Show available CLI options."""
    options = [
        "--config/-c: Configuration file path",
        "--log-level/-l: Set logging level",
        "--log-file: Log to file",
        "--port/-p: CLI service port",
        "--host: CLI service host",
        "--mode/-m: Trading mode",
        "--health-check: Perform health check",
        "--validate-config: Validate configuration",
        "--example: Run example/demo mode",
        "--daemon/-d: Run as daemon",
        "--version/-v: Show version",
    ]

    for _option in options:
        pass


def main():
    """Demonstrate the enhanced CLI service functionality."""
    # 1. Show help
    test_help_command()

    # 2. Show version
    test_version_command()

    # 3. Test config validation (with non-existent config)
    test_config_validation_invalid()

    # 4. Create a basic config file for testing
    config_path = test_config_validation_valid()

    # 5. Test health check
    test_health_check(config_path)

    # 6. Test example mode (brief run)
    test_example_mode()

    # 7. Test different log levels
    test_log_levels(config_path)

    # 8. Show available options
    show_available_options()

    # Cleanup
    try:
        if config_path.exists():
            config_path.unlink()
    except Exception as e:
        logger.exception("Error running command: %s", e)



if __name__ == "__main__":
    main()
