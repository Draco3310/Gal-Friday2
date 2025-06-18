#!/usr/bin/env python3
"""Example demonstrating the enhanced CLI service with comprehensive main guard.

This example shows how to use the CLI service with:
- Command line argument parsing
- Configuration validation
- Health checks
- Different execution modes
- Graceful shutdown
"""

from pathlib import Path
import subprocess
import sys


def run_cli_command(args: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a CLI command and return the result."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "gal_friday.cli_service", *args],
            check=False, capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def main():
    """Demonstrate the enhanced CLI service functionality."""
    # 1. Show help
    returncode, stdout, stderr = run_cli_command(["--help"])
    if returncode == 0:
        for line in stdout.split("\n")[:10]:
            pass
    else:
        pass

    # 2. Show version
    returncode, stdout, stderr = run_cli_command(["--version"])
    if returncode == 0:
        pass
    else:
        pass

    # 3. Test config validation (with non-existent config)
    returncode, stdout, stderr = run_cli_command([
        "--validate-config",
        "--config", "non_existent_config.yaml",
    ])
    if returncode != 0:
        pass
    else:
        pass

    # 4. Create a basic config file for testing
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

    except Exception:
        pass

    # 6. Test health check
    returncode, stdout, stderr = run_cli_command([
        "--health-check",
        "--config", str(config_path) if config_path.exists() else "config/default.yaml",
    ])
    if returncode == 0:
        pass
    else:
        pass

    # 7. Test example mode (brief run)
    returncode, stdout, stderr = run_cli_command(["--example"], timeout=5)
    if returncode in (0, -1):  # Timeout is expected
        if stdout:
            for line in stdout.split("\n")[:5]:
                if line.strip():
                    pass
    else:
        pass

    # 8. Test different log levels
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

    # 9. Show available options
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

    # Cleanup
    try:
        if config_path.exists():
            config_path.unlink()
    except Exception:
        pass



if __name__ == "__main__":
    main()
