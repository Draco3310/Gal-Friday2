# Project Gal-Friday2

Automated cryptocurrency trading bot (High-Frequency Scalping/Day Trading) using AI/ML prediction models, with a focus on backtesting capabilities.

## Goal

Develop, deploy, and maintain a sophisticated, automated cryptocurrency trading bot targeting XRP/USD and DOGE/USD on Kraken, aiming for significant income generation ($75k/year target on $100k capital) within defined risk parameters.

## Current Status

As of May 2025, the project has made significant progress with the implementation of:
- Core architecture with event-driven design
- Backtesting engine for strategy validation
- Multiple service components including portfolio management, risk management, and execution handling
- Configuration management system
- Modular design to support both live trading and simulated environments

## Technology Stack

*   Python 3.9+ with extensive use of `asyncio` for asynchronous operations
*   PostgreSQL for persistent storage
*   Event-driven architecture with a custom publish-subscribe system
*   XGBoost and other ML models for prediction
*   Kraken API integration
*   Key Libraries: `pandas`, `numpy`, `scikit-learn`, `talib`, `websockets`, `aiohttp`, `asyncpg`, `PyYAML`

## Repository Structure

- `src/gal_friday/` - Core trading system components
  - `core/` - Core event system and communication primitives
  - `execution/` - Order execution components
- `config/` - Configuration files and templates
- `docs/` - Comprehensive documentation
  - `Phase 1/` - Requirements analysis and planning
  - `Phase 3/` - Design refinements and code review plans
  - `KrakenAPI/` - API documentation for Kraken integration
- `db/schema/` - Database schema definitions
- `tests/` - Test suite for components
- `scripts/` - Utility scripts including model training

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Draco3310/Gal-Friday2.git
    cd Gal-Friday2
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment:**
    - Copy `config/config.yaml` to create your local configuration
    - Set up PostgreSQL if needed for persistent storage
    - Configure your Kraken API credentials (for live trading)

## Usage

### Backtesting

The system includes a comprehensive backtesting engine to validate strategies before deploying them in a live environment:

```python
# Example backtesting usage
from gal_friday.config_manager import ConfigManager
from gal_friday.backtesting_engine import BacktestingEngine
import asyncio

async def run_backtest():
    config = ConfigManager(config_path="config/config.yaml")
    engine = BacktestingEngine(config)
    results = await engine.run_backtest()
    print(f"Backtest completed with results: {results}")

if __name__ == "__main__":
    asyncio.run(run_backtest())
```

### Live Trading

Live trading functionality requires proper configuration and risk management settings:

```python
# This example shows how to initialize the system for live trading
# Note: Always start with small amounts and thorough testing
from gal_friday.main import setup_and_run_trading_system
import asyncio

if __name__ == "__main__":
    asyncio.run(setup_and_run_trading_system())
```

## Development

The project is currently in active development. See `docs/post_mvp_plan_gal_friday_v0.1.md` for upcoming features and improvements.

## License

[Specify your license information here]

## Acknowledgments

- Kraken for providing the API infrastructure
- Open source community for the various libraries and tools used in this project

# TESTING
