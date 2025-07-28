#!/bin/bash
set -euo pipefail

source .venv/bin/activate

# Run tests with coverage by default
pytest tests/ \
    --cov=gal_friday \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-config=pyproject.toml \
    "$@"

# Open coverage report if requested
if [[ "${OPEN_COVERAGE:-false}" == "true" ]]; then
    open htmlcov/index.html || xdg-open htmlcov/index.html
fi
