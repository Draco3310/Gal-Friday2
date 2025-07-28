# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gal-Friday2 is an enterprise-grade automated cryptocurrency trading bot for high-frequency scalping and day trading on Kraken exchange. It uses ML models (XGBoost, RandomForest, LSTM) for price prediction and implements sophisticated risk management.

## Common Development Commands

### Setup
```bash
# Full development setup
./setup.sh --dev

# Or manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
```

### Running
```bash
# Main application
python -m gal_friday.main --config config/config.yaml

# With debug logging
python -m gal_friday.main --config config/config.yaml --log-level DEBUG

# Monitoring dashboard
python run_dashboard.py
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=gal_friday tests/

# Run specific categories
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m "not slow"  # Skip slow tests
```

### Code Quality
```bash
# All linting
make lint

# Individual tools
make ruff          # Format and lint
make mypy-organized # Type checking
make bandit        # Security scan

# Quick fixes
ruff check --fix gal_friday/
ruff format gal_friday/
```

## Code Quality Standards

### Ruff Configuration
This project uses **ruff v0.12.5** for linting and formatting. Version alignment is critical for consistent results.

**Version Requirements:**
- **Local development**: ruff==0.12.5 (requirements-dev.txt)
- **Pre-commit hooks**: ruff v0.12.5 (.pre-commit-config.yaml)
- **CI/CD pipeline**: ruff>=0.12.0,<0.13.0 (pyproject.toml)

**Version Alignment Issues:**
Previous version misalignment (local v0.8.6 vs pre-commit v0.12.5) caused 80% of configured rules to be silently ignored locally while enforced in pre-commit, leading to inconsistent error reporting.

### Code Quality Governance
Comprehensive governance framework established in `docs/guides/code_quality_governance.md`:

**# noqa Standards:**
```python
# Format: # noqa: <error_code> - <category>: <justification>
from module import item  # noqa: PLC0415 - architectural: prevents circular dependency
```

**Categories:**
- `architectural`: Circular imports, lazy loading for optional dependencies
- `test-complexity`: Acceptable complexity in comprehensive integration tests
- `demo-pattern`: Educational/demonstration code patterns
- `performance`: Performance-critical optimizations

**Quality Gates:**
- Maximum 5 # noqa comments per file
- >95% error resolution rate maintained
- All architectural exceptions documented with detailed justifications
- Expert validation for complex architectural decisions

### Error Resolution Achievement
- **Baseline**: 80 ruff errors (100%)
- **Final**: 0 errors (100% resolution) ✅
- **Strategy**: Version alignment + systematic error categorization and resolution
- **Expert Validation**: O3 and Gemini model consensus on architectural patterns

## Architecture Overview

### Event-Driven Architecture
The system uses an asynchronous event bus pattern where modules communicate via events:

1. **DataIngestor** → MarketDataEvent → Feature extraction
2. **FeatureEngine** → FeatureEvent → ML prediction
3. **PredictionService** → PredictionEvent → Strategy decision
4. **StrategyArbitrator** → TradeSignalProposedEvent → Risk check
5. **RiskManager** → TradeSignalApprovedEvent → Order execution
6. **ExecutionHandler** → ExecutionReportEvent → Portfolio update

### Key Modules
- **core/**: Event definitions, types, pubsub implementation
- **dal/**: Data access layer with repository pattern
- **execution/**: Kraken API integration (REST + WebSocket)
- **market_price/**: Real-time market data handling
- **model_lifecycle/**: ML model training and management
- **predictors/**: ML prediction implementations
- **portfolio/**: Position and balance tracking
- **monitoring/**: FastAPI dashboard and metrics

### Data Flow
- **Market Data**: WebSocket → DataIngestor → InfluxDB/PostgreSQL
- **Features**: Raw data → FeatureEngine → In-memory cache
- **Predictions**: Features → ML models (ProcessPool) → Event bus
- **Orders**: Risk-approved signals → ExecutionHandler → Kraken API

### Risk Management
- Max drawdown: 15% (triggers HALT)
- Daily limit: 2%, Weekly: 5%
- Risk per trade: 0.5-1% of equity
- Circuit breakers for API failures

### Database Architecture
- **PostgreSQL**: Trades, positions, logs, configuration
- **InfluxDB**: Time-series market data and predictions
- **SQLite**: Development/testing fallback

### Important Patterns
- **Async-first**: All I/O operations use asyncio
- **Event sourcing**: State changes tracked via events
- **Repository pattern**: Clean data access separation
- **Strategy pattern**: Pluggable ML models and strategies
- **Circuit breaker**: API resilience with exponential backoff

# Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive
context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

## File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the
  gemini command:

### Examples:

**Single file analysis:**
gemini -p "@src/main.py Explain this file's purpose and structure"

Multiple files:
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

Entire directory:
gemini -p "@src/ Summarize the architecture of this codebase"

Multiple directories:
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

Current directory and subdirectories:
gemini -p "@./ Give me an overview of this entire project"

# Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and dependencies"

Implementation Verification Examples


Check if a feature is implemented:
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

Verify authentication implementation:
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

Check for specific patterns:
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

Verify error handling:
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

Check for rate limiting:
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

Verify caching strategy:
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

Check for specific security measures:
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

Verify test coverage for features:
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"

When to Use Gemini CLI

Use gemini -p when:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

Important Notes

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results

## Code Quality Procedures

### Before Making Code Changes
1. **Version Check**: Ensure ruff v0.12.5 is installed locally
2. **Pre-commit Setup**: Verify pre-commit hooks are properly configured
3. **Baseline Check**: Run `ruff check gal_friday/` to establish current error count

### During Development
1. **Incremental Linting**: Run ruff checks frequently during development
2. **Auto-fixes First**: Use `ruff check --fix` for automatic corrections
3. **Manual Review**: Address remaining errors through code improvements or documented # noqa

### Error Resolution Protocol
1. **Simple Fixes**: Prefer code improvements over # noqa when possible
2. **Architectural Necessity**: Use # noqa with detailed justification for circular imports
3. **Test Complexity**: Document acceptable complexity in comprehensive integration tests
4. **Governance Review**: Reference `docs/guides/code_quality_governance.md` for standards

### Quality Assurance
1. **Local Validation**: Ensure local ruff reports match pre-commit results
2. **Documentation**: Update governance docs when introducing new # noqa patterns
3. **Expert Consensus**: For complex architectural decisions, seek validation from multiple models
4. **Version Alignment**: Maintain consistent ruff versions across all environments

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
