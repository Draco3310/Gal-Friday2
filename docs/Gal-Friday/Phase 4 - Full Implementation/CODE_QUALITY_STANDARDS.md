# Code Quality Standards

This document outlines the code quality standards and tools used in the Gal-Friday2 project. These standards have been implemented to ensure consistency, maintainability, and reliability of the codebase.

## Tools Overview

The following tools are used for enforcing code quality:

### Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter and formatter, written in Rust. It replaces multiple tools (flake8, isort, Black) with a single unified tool.

- **Purpose**: Linting and formatting Python code
- **Configuration**: Defined in `pyproject.toml` under `[tool.ruff]` sections
- **Key Features**:
  - Fast execution (10-100x faster than other tools)
  - Comprehensive rule set covering style, bugs, complexity
  - Automatic fixing capabilities
  - Google-style docstring enforcement

### Mypy

[Mypy](https://mypy.readthedocs.io/) is a static type checker for Python.

- **Purpose**: Verify type annotations and catch type-related errors
- **Configuration**: Defined in `pyproject.toml` under `[tool.mypy]` sections
- **Key Features**:
  - Enforces type annotations on functions and methods
  - Different strictness levels for different parts of the codebase
  - Relaxed settings for test files

### Bandit

[Bandit](https://github.com/PyCQA/bandit) is a tool designed to find common security issues in Python code.

- **Purpose**: Identify potential security vulnerabilities
- **Configuration**: Defined in `pyproject.toml` under `[tool.bandit]` section
- **Key Features**:
  - Scans for common security issues like hardcoded passwords
  - Identifies use of insecure functions and modules

### Pre-commit

[Pre-commit](https://pre-commit.com/) is a framework for managing git pre-commit hooks.

- **Purpose**: Run code quality checks automatically before commits
- **Configuration**: Defined in `.pre-commit-config.yaml`
- **Key Features**:
  - Integrates all the above tools
  - Prevents committing code that doesn't meet standards
  - Can be configured to be non-blocking for specific hooks

## Code Style Guidelines

### Python Style

- Follow [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Maximum line length: 99 characters
- Use Google-style docstrings for all public functions, classes, and methods

### Type Annotations

- Required for all function parameters and return values
- Required for class attributes
- Exceptions are allowed in test files

### Documentation

- All modules should have module-level docstrings
- All public classes and functions should have docstrings
- Complex logic should be commented
- Inline `# noqa` comments should include the specific rule being ignored

## Testing Standards

- All new features must include tests
- Test coverage should be maintained at 80% or higher
- Test categories include:
  - Unit tests
  - Integration tests
  - Slow-running tests (marked appropriately)
  - Memory profiling tests

## Continuous Integration

The project uses GitHub Actions for continuous integration, with workflows defined in `.github/workflows/`:

- **Code Quality Checks**: Run on every push and pull request
- **Test Suite**: Executes the full test suite with coverage reporting

## Getting Started with Code Quality Tools

### Local Setup

1. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tools Manually

- **Ruff Linting**:
  ```bash
  ruff check gal_friday/
  ```

- **Ruff Formatting**:
  ```bash
  ruff format gal_friday/
  ```

- **Mypy Type Checking**:
  ```bash
  mypy
  ```

- **Bandit Security Scanning**:
  ```bash
  bandit -r gal_friday
  ```

- **All Pre-commit Hooks**:
  ```bash
  pre-commit run --all-files
  ```

## Updating Code Quality Tools

When updating or changing code quality tools:

1. Update `pyproject.toml` with new configuration
2. Update `.pre-commit-config.yaml` if necessary
3. Update `requirements.txt` to include new dependencies
4. Run all tools manually to verify changes
5. Document changes in release notes

## Exceptions and Special Cases

- Generated code may be excluded from some checks
- Legacy code being refactored can have exceptions with proper justification
- Performance-critical sections may have relaxed rules in specific cases
