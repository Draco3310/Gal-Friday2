#!/bin/bash
set -euo pipefail

# Load environment
source .venv/bin/activate
source .env

# Set Python path
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Run with optional arguments
exec python -m gal_friday.main --config config/config.yaml "$@"
