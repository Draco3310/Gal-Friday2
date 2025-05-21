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
   pip install pytest pytest-cov flake8 flake8-docstrings mypy black bandit pylint isort pydocstyle pre-commit
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style and Quality Tools

We use several tools to ensure code quality and consistency:

### Configuration Files

- **pyproject.toml**: Primary configuration file for:
  - Black (code formatting)
  - isort (import sorting)
  - mypy (static type checking)
  - pytest (testing)
  - bandit (security linting)

- **.flake8**: Configuration for flake8 (style guide enforcement)

- **.pre-commit-config.yaml**: Configuration for pre-commit hooks

### Running Code Quality Checks

You can run the following commands to check your code:

```bash
# Code formatting
black src tests

# Import sorting
isort src tests

# Type checking
mypy tests src

# Style guide enforcement
flake8 tests src

# Run all pre-commit checks
pre-commit run --all-files
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
pytest --cov=src tests/

# Run specific test file
pytest tests/path/to/test_file.py
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
├── src/
│   └── gal_friday/  # Main package code
├── tests/
│   ├── unit/        # Unit tests
│   └── integration/ # Integration tests
├── docs/            # Documentation
├── scripts/         # Utility scripts
└── config/          # Configuration files
``` 