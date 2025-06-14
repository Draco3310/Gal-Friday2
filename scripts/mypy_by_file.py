#!/usr/bin/env python3
"""
Organize MyPy output by file for better readability.
Usage: python scripts/mypy_by_file.py [mypy_args...]
"""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_mypy_organized(args=None):
    """Run mypy and organize output by file."""
    if args is None:
        args = sys.argv[1:] if len(sys.argv) > 1 else ["gal_friday/"]
    
    # Run mypy and capture output
    try:
        result = subprocess.run(
            ["mypy"] + args,
            capture_output=True,
            text=True,
            check=False
        )
    except FileNotFoundError:
        print("ERROR: MyPy not found. Install with: pip install mypy")
        return 1
    
    output_lines = result.stdout.split('\n') + result.stderr.split('\n')
    output_lines = [line for line in output_lines if line.strip()]
    
    # Organize by file
    file_errors = defaultdict(list)
    summary_lines = []
    
    for line in output_lines:
        if ':' in line and any(line.startswith(prefix) for prefix in ['gal_friday/', './', '/']):
            # Extract filename from error line
            parts = line.split(':', 2)
            if len(parts) >= 2:
                filename = parts[0]
                error_detail = ':'.join(parts[1:])
                file_errors[filename].append(error_detail)
            else:
                summary_lines.append(line)
        else:
            summary_lines.append(line)
    
    # Print organized output
    if file_errors:
        print("MyPy Results Organized by File")
        print("=" * 60)
        
        for filename in sorted(file_errors.keys()):
            errors = file_errors[filename]
            print(f"\n[FILE] {filename} ({len(errors)} issues)")
            print("-" * (len(filename) + 20))
            
            for error in errors:
                print(f"  {error}")
        
        print(f"\nSummary:")
        print(f"   Files with issues: {len(file_errors)}")
        print(f"   Total issues: {sum(len(errors) for errors in file_errors.values())}")
    
    # Print any summary lines
    if summary_lines:
        non_empty_summary = [line for line in summary_lines if line.strip()]
        if non_empty_summary:
            print(f"\nMyPy Summary:")
            for line in non_empty_summary:
                print(f"   {line}")
    
    # Print success message if no errors
    if not file_errors and not any(line.strip() for line in summary_lines):
        print("SUCCESS: No MyPy errors found!")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_mypy_organized())