#!/bin/bash
set -euo pipefail

source .venv/bin/activate

echo "Running code quality checks..."

# Ruff
echo "Running ruff..."
ruff check gal_friday/ tests/ --fix

# Black
echo "Running black..."
black gal_friday/ tests/

# isort
echo "Running isort..."
isort gal_friday/ tests/

# MyPy
echo "Running mypy..."
mypy gal_friday/

# Bandit
echo "Running bandit security scan..."
bandit -r gal_friday/ -f json -o .bandit-report.json || true

# Safety
echo "Checking dependencies for vulnerabilities..."
safety check --json || true

echo "Code quality checks complete!"
