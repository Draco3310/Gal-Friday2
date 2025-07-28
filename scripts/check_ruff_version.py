#!/usr/bin/env python3
"""Ruff version validation script.

This script ensures that the local ruff version matches the project's requirements.
Used in development workflows and CI/CD pipelines to prevent version-related
linting discrepancies.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_installed_ruff_version() -> Optional[str]:
    """Get the currently installed ruff version."""
    try:
        result = subprocess.run(
            ["ruff", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output format: "ruff 0.12.5"
        return result.stdout.strip().split()[-1]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def get_required_ruff_version() -> Optional[str]:
    """Get the required ruff version from requirements-dev.txt."""
    try:
        requirements_file = Path(__file__).parent.parent / "requirements-dev.txt"
        with requirements_file.open() as f:
            for line in f:
                line = line.strip()
                if line.startswith("ruff=="):
                    return line.split("==")[1]
    except (FileNotFoundError, IndexError):
        pass
    return None


def check_ruff_version() -> bool:
    """Check if installed ruff version matches requirements.

    Returns:
        True if versions match, False otherwise.
    """
    installed = get_installed_ruff_version()
    required = get_required_ruff_version()

    if installed is None:
        print("❌ Error: ruff is not installed or not in PATH", file=sys.stderr)
        return False

    if required is None:
        print("❌ Error: Could not find required ruff version in requirements-dev.txt", file=sys.stderr)
        return False

    if installed != required:
        print(f"❌ Version mismatch:", file=sys.stderr)
        print(f"   Installed: ruff {installed}", file=sys.stderr)
        print(f"   Required:  ruff {required}", file=sys.stderr)
        print(f"   Fix with:  pip install ruff=={required}", file=sys.stderr)
        return False

    print(f"✅ Ruff version check passed: {installed}")
    return True


def main() -> None:
    """Main entry point."""
    if check_ruff_version():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
