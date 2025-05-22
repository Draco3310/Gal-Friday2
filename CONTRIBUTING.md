# Contributing to Gal-Friday2

Thank you for your interest in contributing to Gal-Friday2! This document provides guidelines and information about the development workflow.

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Gal-Friday2.git
   cd Gal-Friday2
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix/MacOS
   source .venv/bin/activate

   pip install -r requirements.txt
   pip install -e .
   ```

3. Install development dependencies:
   ```bash
   pip install pytest pytest-cov pytest-asyncio mypy ruff bandit pre-commit memory_profiler
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style and Quality Tools

We use several tools to ensure code quality and consistency:

### Configuration Files

- **pyproject.toml**: Primary configuration file for:
  - Ruff (linting, formatting, and import sorting)
  - mypy (static type checking)
  - pytest (testing)
  - bandit (security scanning)
  - pip-compile (dependency management)

- **.pre-commit-config.yaml**: Configuration for pre-commit hooks

### Running Code Quality Checks

You can run the following commands to check your code:

```bash
# Linting and code formatting with Ruff
ruff check --fix gal_friday tests

# Format code with Ruff
ruff format gal_friday tests

# Type checking
mypy gal_friday tests

# Security scanning
bandit -r gal_friday

# Run all pre-commit checks
pre-commit run --all-files

# Memory profiling for critical modules
python -m memory_profiler gal_friday/execution_handler.py
```

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Make your changes, including tests
3. Ensure all checks pass (pre-commit, tests)
4. Submit a pull request

## Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=gal_friday --cov-report=term --cov-report=xml:coverage.xml

# Run specific test categories
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m slow          # Only slow tests

# Run specific test file
pytest tests/path/to/test_file.py

# Run tests in parallel
pytest -xvs
```

## CI/CD Pipeline

Our GitHub Actions workflow performs the following checks on every pull request and push to main branches:

1. Runs pre-commit hooks
2. Runs tests with pytest
3. Builds the package
4. Builds the Docker image
5. Deploys to the development environment (for pushes to main/master)

Make sure your code passes all these checks before submitting a pull request.

## Project Structure

```
Gal-Friday2/
├── gal_friday/              # Main package code
│   ├── core/                # Core components and abstractions
│   ├── predictors/          # Prediction models (LSTM, ensemble, etc.)
│   ├── execution/           # Exchange execution handlers
│   └── strategies/          # Trading strategies
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── .github/                 # GitHub configuration
│   └── workflows/           # GitHub Actions workflows
├── config/                  # Configuration files
└── data/                    # Data storage
```

## CI/CD and Code Quality

The project uses GitHub Actions for continuous integration with detailed reporting on:

- Code quality (Ruff linting/formatting)
- Type checking (Mypy)
- Security vulnerabilities (Bandit)
- Test coverage
- Memory profiling for performance-critical components
- Dependency security checks

These reports are available in the GitHub Actions workflow summaries.
