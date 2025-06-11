# AGENTS.md - Gal-Friday2 Project Configuration

This file provides OpenAI's Codex with comprehensive guidance for working with the Gal-Friday2 cryptocurrency trading bot project.

## Project Overview

Gal-Friday2 is an automated cryptocurrency trading bot designed for high-frequency scalping and day trading on the Kraken exchange. It uses AI/ML predictive models (XGBoost, RandomForest, LSTM) to trade XRP/USD and DOGE/USD pairs with comprehensive risk management and real-time monitoring.

## Project Structure and Conventions

### Directory Structure
```
Gal-Friday2/
├── gal_friday/                 # Main package code
│   ├── core/                   # Core components and abstractions
│   ├── dal/                    # Data Access Layer (models, repositories, migrations)
│   ├── execution/              # Order execution and exchange adapters
│   ├── interfaces/             # Abstract base classes and interfaces
│   ├── market_price/           # Market data services
│   ├── model_lifecycle/        # ML model management
│   ├── model_training/         # Model training pipelines
│   ├── models/                 # Data models and schemas
│   ├── monitoring/             # System monitoring and alerting
│   ├── portfolio/              # Portfolio and position management
│   ├── predictors/             # ML prediction models
│   ├── providers/              # Data providers (API, database, file)
│   ├── services/               # Business logic services
│   └── utils/                  # Utility functions and helpers
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── dal/                    # Database-specific tests
├── config/                     # Configuration files
├── db/                         # Database schemas and migrations
├── docs/                       # Project documentation
├── scripts/                    # Utility and deployment scripts
└── examples/                   # Example code and usage
```

### File Naming Conventions
- **Python files**: Use snake_case (e.g., `market_price_service.py`)
- **Classes**: Use PascalCase (e.g., `MarketPriceService`)
- **Functions/methods**: Use snake_case (e.g., `get_latest_price`)
- **Constants**: Use UPPER_SNAKE_CASE (e.g., `MAX_RETRY_ATTEMPTS`)
- **Private methods**: Prefix with underscore (e.g., `_internal_method`)

### Import Organization
Follow this import order (enforced by Ruff):
1. Standard library imports
2. Third-party imports
3. Local application imports

Use absolute imports for the main package:
```python
from gal_friday.core.event_store import EventStore
from gal_friday.models.order import Order
```

## Coding Style Guidelines

### Code Formatting
- **Line length**: Maximum 99 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Trailing commas**: Required in multi-line structures

### Type Annotations
- **Required**: All function signatures must have type annotations
- **Return types**: Always specify return types
- **Complex types**: Use `typing` module for complex types
- **Optional types**: Use `| None` syntax (Python 3.10+ union syntax)

Example:
```python
from typing import Dict, List, Optional
from decimal import Decimal

async def get_portfolio_value(
    self,
    trading_pairs: List[str],
    include_unrealized: bool = True
) -> Decimal | None:
    """Get total portfolio value."""
    pass
```

### Docstring Style
Use Google-style docstrings:
```python
def calculate_position_size(
    self,
    account_balance: Decimal,
    risk_percentage: Decimal
) -> Decimal:
    """Calculate position size based on risk parameters.
    
    Args:
        account_balance: Current account balance in base currency.
        risk_percentage: Risk percentage (0.01 = 1%).
        
    Returns:
        Calculated position size in base currency.
        
    Raises:
        ValueError: If risk_percentage is negative or > 1.
    """
    pass
```

### Error Handling
- Use specific exception types from `gal_friday.exceptions`
- Always log errors with context
- Use try/except blocks for external API calls
- Implement retry logic for transient failures

### Async/Await Patterns
- Use `async`/`await` for I/O operations
- Prefer `asyncio.gather()` for concurrent operations
- Use `asyncio.create_task()` for fire-and-forget operations
- Always handle `asyncio.CancelledError`

## Test and Linting Commands

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gal_friday --cov-report=term --cov-report=xml:coverage.xml

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only

# Run specific test file
pytest tests/unit/test_market_price_service.py

# Run tests in parallel with verbose output
pytest -xvs

# Run tests with memory profiling
pytest -m memory_profile
```

### Code Quality Checks
```bash
# Linting and auto-fix with Ruff
ruff check --fix gal_friday tests

# Code formatting with Ruff
ruff format gal_friday tests

# Type checking with mypy
mypy gal_friday tests

# Security scanning with bandit
bandit -r gal_friday

# Run all pre-commit checks
pre-commit run --all-files

# Memory profiling for performance-critical modules
python -m memory_profiler gal_friday/execution_handler.py
```

### Validation Checks (REQUIRED)
After making any code changes, Codex MUST run these validation checks:

```bash
# 1. Format code
ruff format gal_friday tests

# 2. Fix linting issues
ruff check --fix gal_friday tests

# 3. Type checking
mypy gal_friday

# 4. Run relevant tests
pytest tests/unit/  # For unit test changes
pytest tests/integration/  # For integration changes
pytest  # For major changes

# 5. Security check
bandit -r gal_friday
```

## Configuration Management

### Environment Variables
- `GAL_FRIDAY_CONFIG_PATH`: Path to configuration file
- `GAL_FRIDAY_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `GAL_FRIDAY_ENV`: Environment (development, staging, production)

### Configuration Files
- Primary config: `config/config.yaml`
- Example config: `config/config.example.yaml`
- Database config: Managed through main config file

## Database Conventions

### Alembic Migrations
- Location: `gal_friday/dal/migrations/`
- Always create migrations for schema changes
- Use descriptive migration names
- Test migrations both up and down

### Model Definitions
- Location: `gal_friday/dal/models/`
- Use SQLAlchemy 2.0+ syntax
- Include proper type annotations
- Add docstrings for complex models

### Repository Pattern
- Location: `gal_friday/dal/repositories/`
- Inherit from `BaseRepository`
- Implement async methods
- Handle database exceptions properly

## Pull Request Conventions

### PR Title Format
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scope: core, execution, monitoring, portfolio, etc.

Examples:
feat(execution): add batch order placement support
fix(monitoring): resolve memory leak in alert service
docs(readme): update installation instructions
```

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings/errors
```

## Specific Instructions

### Working with Financial Data
- Always use `Decimal` for monetary values, never `float`
- Include proper precision handling for different currencies
- Validate all financial calculations
- Log all financial operations for audit trails

### Exchange Integration
- Always implement proper rate limiting
- Handle API errors gracefully with retries
- Validate all order parameters before submission
- Implement proper WebSocket reconnection logic

### Machine Learning Models
- Version all model artifacts
- Include model metadata and performance metrics
- Implement proper model validation
- Handle model loading failures gracefully

### Risk Management
- Validate all risk parameters before trading
- Implement circuit breakers for abnormal conditions
- Log all risk decisions with context
- Never bypass risk checks in production code

### Monitoring and Alerting
- Include structured logging with correlation IDs
- Implement health checks for all critical components
- Set up proper alerting thresholds
- Monitor system resources and performance

## Security Guidelines

### API Keys and Secrets
- Never commit API keys or secrets to version control
- Use environment variables or secure vaults
- Implement proper key rotation procedures
- Log access to sensitive operations (without exposing secrets)

### Data Protection
- Encrypt sensitive data at rest
- Use secure connections for all external communications
- Implement proper access controls
- Audit all data access operations

## Performance Considerations

### Memory Management
- Profile memory usage for long-running processes
- Implement proper cleanup for large datasets
- Use generators for processing large data streams
- Monitor memory leaks in production

### Async Operations
- Use connection pooling for database operations
- Implement proper timeout handling
- Avoid blocking operations in async contexts
- Use appropriate concurrency limits

## Troubleshooting Common Issues

### Database Connection Issues
- Check connection pool settings
- Verify database credentials and network connectivity
- Review migration status
- Check for long-running transactions

### Exchange API Issues
- Verify API credentials and permissions
- Check rate limiting and retry logic
- Review WebSocket connection status
- Validate order parameters and market conditions

### Model Performance Issues
- Check model loading and initialization
- Verify feature engineering pipeline
- Review prediction latency and accuracy
- Monitor model drift and performance degradation

## Development Workflow

1. **Create feature branch** from main
2. **Implement changes** following coding standards
3. **Write/update tests** for new functionality
4. **Run validation checks** (required)
5. **Update documentation** if needed
6. **Submit pull request** with proper description
7. **Address review feedback**
8. **Merge after approval**

## Contact and Support

For questions about this configuration or the project:
- Review existing documentation in `docs/`
- Check the issue tracker for known problems
- Refer to `CONTRIBUTING.md` for detailed guidelines
- Review the `README.md` for project overview

---

**Note**: This AGENTS.md file applies to the entire Gal-Friday2 project tree. Always follow these guidelines when making changes to ensure consistency and maintainability.
