#!/usr/bin/env python3
"""
Example demonstrating the enhanced CLI service with comprehensive main guard.

This example shows how to use the CLI service with:
- Command line argument parsing
- Configuration validation
- Health checks
- Different execution modes
- Graceful shutdown
"""

import subprocess
import sys
import time
from pathlib import Path


def run_cli_command(args: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a CLI command and return the result."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "gal_friday.cli_service"] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def main():
    """Demonstrate the enhanced CLI service functionality."""
    
    print("=" * 60)
    print("CLI SERVICE ENHANCED MAIN GUARD DEMONSTRATION")
    print("=" * 60)
    
    # 1. Show help
    print("\n1. Displaying CLI help...")
    returncode, stdout, stderr = run_cli_command(["--help"])
    if returncode == 0:
        print("✅ Help command successful")
        print("Help output (first 10 lines):")
        for line in stdout.split('\n')[:10]:
            print(f"   {line}")
        print("   ...")
    else:
        print(f"❌ Help command failed: {stderr}")
    
    # 2. Show version
    print("\n2. Checking version...")
    returncode, stdout, stderr = run_cli_command(["--version"])
    if returncode == 0:
        print(f"✅ Version: {stdout.strip()}")
    else:
        print(f"❌ Version command failed: {stderr}")
    
    # 3. Test config validation (with non-existent config)
    print("\n3. Testing configuration validation with non-existent file...")
    returncode, stdout, stderr = run_cli_command([
        "--validate-config", 
        "--config", "non_existent_config.yaml"
    ])
    if returncode != 0:
        print("✅ Correctly detected missing configuration file")
    else:
        print("❌ Should have failed for missing config file")
    
    # 4. Create a basic config file for testing
    print("\n4. Creating test configuration file...")
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
        print("✅ Test configuration file created")
        
        # Test config validation with valid file
        print("\n5. Testing configuration validation with valid file...")
        returncode, stdout, stderr = run_cli_command([
            "--validate-config",
            "--config", str(config_path)
        ])
        if returncode == 0:
            print("✅ Configuration validation successful")
        else:
            print(f"❌ Configuration validation failed: {stderr}")
        
    except Exception as e:
        print(f"❌ Error creating test config: {e}")
    
    # 6. Test health check
    print("\n6. Testing health check...")
    returncode, stdout, stderr = run_cli_command([
        "--health-check",
        "--config", str(config_path) if config_path.exists() else "config/default.yaml"
    ])
    if returncode == 0:
        print("✅ Health check passed")
    else:
        print(f"⚠️  Health check issues detected (expected): {stderr}")
    
    # 7. Test example mode (brief run)
    print("\n7. Testing example/demo mode (3 seconds)...")
    returncode, stdout, stderr = run_cli_command(["--example"], timeout=5)
    if returncode == 0 or returncode == -1:  # Timeout is expected
        print("✅ Example mode started successfully")
        if stdout:
            print("Sample output:")
            for line in stdout.split('\n')[:5]:
                if line.strip():
                    print(f"   {line}")
    else:
        print(f"❌ Example mode failed: {stderr}")
    
    # 8. Test different log levels
    print("\n8. Testing different log levels...")
    for log_level in ["DEBUG", "WARNING", "ERROR"]:
        print(f"   Testing log level: {log_level}")
        returncode, stdout, stderr = run_cli_command([
            "--validate-config",
            "--log-level", log_level,
            "--config", str(config_path) if config_path.exists() else "config/default.yaml"
        ])
        if returncode == 0:
            print(f"   ✅ {log_level} log level working")
        else:
            print(f"   ❌ {log_level} log level failed")
    
    # 9. Show available options
    print("\n9. Available CLI options:")
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
        "--version/-v: Show version"
    ]
    
    for option in options:
        print(f"   • {option}")
    
    # Cleanup
    print("\n10. Cleaning up...")
    try:
        if config_path.exists():
            config_path.unlink()
            print("✅ Test configuration file cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("CLI SERVICE DEMONSTRATION COMPLETED")
    print("=" * 60)
    print("\n📚 Usage Examples:")
    print("   python -m gal_friday.cli_service --example")
    print("   python -m gal_friday.cli_service --health-check")
    print("   python -m gal_friday.cli_service --config config/prod.yaml --log-level DEBUG")
    print("   python -m gal_friday.cli_service --help")


if __name__ == "__main__":
    main() 