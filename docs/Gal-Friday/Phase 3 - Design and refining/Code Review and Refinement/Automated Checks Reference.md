# Automated Checks Reference Guide

This document provides instructions for running the remaining automated code quality checks and setting up pre-commit hooks to automate these checks.

## 1. Running Pylint

Pylint performs comprehensive static analysis beyond what flake8 covers, generating a score and identifying potential code quality issues.

### Installation (if not already installed)
```bash
pip install pylint
```

### Configuration
Ensure `pyproject.toml` contains the pylint configuration:

```toml
[tool.pylint]
disable = [
    "C0111",  # missing-docstring
    "C0103",  # invalid-name
    "C0330",  # bad-continuation
    "C0326",  # bad-whitespace
    "W0511",  # fixme
    "R0903",  # too-few-public-methods
    "R0913",  # too-many-arguments
    "R0914",  # too-many-locals
]
ignore = [
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
]
good-names = ["i", "j", "k", "ex", "Run", "_", "id"]
max-line-length = 100
```

### Running Pylint
```bash
# Run on the entire package
pylint src/gal_friday

# Run on specific files or directories
pylint src/gal_friday/prediction_service.py
pylint src/gal_friday/predictors/
```

### Interpreting Results
- Pylint provides a score from 0-10 where 10 is perfect
- The output includes categorized messages:
  - C: Convention (style)
  - R: Refactor (complexity)
  - W: Warning (potential problems)
  - E: Error (likely bugs)
  - F: Fatal (critical)

## 2. Running Bandit

Bandit checks for common security vulnerabilities in Python code.

### Installation (if not already installed)
```bash
pip install bandit
```

### Configuration
Create a `.bandit` file in the project root:

```yaml
skips: ['B101', 'B403']  # Example skips: assertions and pickle
```

### Running Bandit
```bash
# Run on entire codebase
bandit -r src/

# Run on specific files or directories
bandit -r src/gal_friday/market_price_service.py

# Run with increased verbosity
bandit -v -r src/
```

### Interpreting Results
Bandit's output includes:
- A summary of the issues found
- Severity level (LOW, MEDIUM, HIGH)
- Confidence level (LOW, MEDIUM, HIGH)
- Line numbers and code snippets with issues

## 3. Running Pydocstyle

Pydocstyle checks that Python docstrings comply with standards (PEP 257 by default).

### Installation (if not already installed)
```bash
pip install pydocstyle
```

### Configuration
Add pydocstyle configuration to `pyproject.toml`:

```toml
[tool.pydocstyle]
convention = "google"  # or "numpy", "pep257"
match = "(?!test_).*\\.py"  # Exclude test files
ignore = [
    "D107",  # Missing docstring in __init__
    "D203",  # 1 blank line required before class docstring
    "D212",  # Multi-line docstring summary should start at the first line
]
```

### Running Pydocstyle
```bash
# Run on entire package
pydocstyle src/gal_friday

# Run on specific files or directories
pydocstyle src/gal_friday/prediction_service.py
```

### Interpreting Results
Pydocstyle outputs information about docstring issues:
- The error code (e.g., D100, D101)
- A description of the issue
- The file and line number

## 4. Setting Up Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit, ensuring code quality standards are maintained.

### Installation (if not already installed)
```bash
pip install pre-commit
```

### Configuration
Create or update `.pre-commit-config.yaml` in the project root:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
    -   id: bandit
        args: ['-ll']  # Report only HIGH and MEDIUM severity issues
        exclude: tests/

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
```

### Installing the Git Hook Scripts
```bash
pre-commit install
```

### Running Against All Files
```bash
pre-commit run --all-files
```

### Running Specific Hooks
```bash
pre-commit run flake8 --all-files
pre-commit run bandit --all-files
```

## 5. Integration with CI/CD

To ensure these checks are run in your CI/CD pipeline, add the following steps to your GitHub workflow file:

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install pylint bandit pydocstyle pre-commit

- name: Run pylint
  run: pylint src/gal_friday

- name: Run bandit
  run: bandit -r src/

- name: Run pydocstyle
  run: pydocstyle src/gal_friday

- name: Run pre-commit hooks
  run: pre-commit run --all-files
```

## 6. Troubleshooting Common Issues

### Pylint
- **False positives**: Disable specific rules in `.pylintrc` or with inline comments `# pylint: disable=rule-code`
- **Import errors**: Check your PYTHONPATH and module structure

### Bandit
- **Performance issues**: Use `--exclude` to skip test directories or large third-party modules
- **False positives**: Use `# nosec` comments to mark false positives

### Pydocstyle
- **Conflicting conventions**: Ensure you're consistently using one docstring style (Google, NumPy, PEP 257)
- **Missing docstrings**: Prioritize public API documentation over internal functions

### Pre-commit
- **Slow hooks**: Move time-consuming hooks (like mypy) to manual runs or CI/CD
- **Hook failures**: Use `--no-verify` to bypass hooks temporarily when needed (use sparingly)
