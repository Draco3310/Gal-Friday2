#!/usr/bin/env python3
"""
Gal Friday Real-Time Trading Dashboard Launcher
Run this script from the project root to start the dashboard
"""

import os
import sys
from pathlib import Path

def main():
    """Launch the dashboard"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    dashboard_script = project_root / "gal_friday" / "monitoring" / "run_dashboard.py"
    
    if not dashboard_script.exists():
        print("❌ Dashboard script not found!")
        print(f"Expected location: {dashboard_script}")
        sys.exit(1)
    
    print("🚀 Starting Gal Friday Real-Time Trading Dashboard...")
    print(f"📁 Project root: {project_root}")
    print(f"🎯 Dashboard script: {dashboard_script}")
    
    # Change to the monitoring directory and run the dashboard
    os.chdir(dashboard_script.parent)
    
    # Execute the dashboard script
    import subprocess
    try:
        subprocess.run([sys.executable, "run_dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 