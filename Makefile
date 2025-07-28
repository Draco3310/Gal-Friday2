.PHONY: lint mypy-organized mypy-summary ruff test clean

# Run all linting tools
lint: ruff mypy-organized bandit

# Run mypy with organized output
mypy-organized:
	@echo "🔍 Running MyPy with file-organized output..."
	@python scripts/mypy_by_file.py gal_friday/

# Run mypy with summary only
mypy-summary:
	@echo "📊 Running MyPy summary..."
	@mypy gal_friday/ --error-summary --tb=short

# Run ruff checks
ruff:
	@echo "🧹 Running Ruff..."
	@ruff check gal_friday/
	@ruff format --check gal_friday/

# Run bandit security checks
bandit:
	@echo "🔒 Running Bandit security scan..."
	@bandit -r gal_friday/ -f json | python -m json.tool

# Run tests
test:
	@echo "🧪 Running tests..."
	@pytest tests/ -v

# Clean cache and temporary files
clean:
	@echo "🧽 Cleaning cache files..."
	@rm -rf .mypy_cache .pytest_cache __pycache__ .coverage htmlcov
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
