#!/usr/bin/env python3
"""Gal Friday Real-Time Trading Dashboard Launcher
Run this script from the project root to start the dashboard.
"""

import os
from pathlib import Path
import sys


def main():
    """Launch the dashboard."""
    # Get the project root directory
    project_root = Path(__file__).parent
    dashboard_script = project_root / "gal_friday" / "monitoring" / "run_dashboard.py"

    if not dashboard_script.exists():
        sys.exit(1)


    # Change to the monitoring directory and run the dashboard
    os.chdir(dashboard_script.parent)

    # Execute the dashboard script
    import subprocess
    try:
        subprocess.run([sys.executable, "run_dashboard.py"], check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
