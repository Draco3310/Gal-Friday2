# Development Guide

## Environment Setup

The `setup.sh` script handles complete environment setup:

```bash
# Development setup (SQLite, minimal deps)
./setup.sh --dev

# Full setup with Docker services
./setup.sh --full --docker

# Test environment
./setup.sh --test
```

## Development Workflow

1. **Activate virtual environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Run tests**
   ```bash
   scripts/dev/test.sh
   ```

3. **Code quality checks**
   ```bash
   scripts/dev/lint.sh
   ```

4. **Run application**
   ```bash
   scripts/dev/run.sh
   ```

## Project Structure

```
gal_friday/
├── core/           # Core types, events, pubsub
├── dal/            # Data access layer
├── execution/      # Order execution
├── market_price/   # Market data handling
├── predictors/     # ML models
├── portfolio/      # Portfolio management
├── monitoring/     # Dashboard and metrics
└── main.py         # Entry point
```

## Coding Standards

- Use type hints for all functions
- Follow PEP 8 style guide
- Write docstrings for all public APIs
- Add unit tests for new features
- Run pre-commit hooks before committing

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gal_friday

# Run specific test
pytest tests/test_module.py::test_function

# Run integration tests
pytest -m integration
```

## Debugging

Use VS Code or PyCharm with provided configurations, or:

```bash
# Debug mode
python -m gal_friday.main --config config/config.yaml --log-level DEBUG

# Profile performance
scripts/dev/profile.sh
```
