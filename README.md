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
- Advanced, modular `PredictionService` supporting multiple, concurrent ML models (XGBoost, Scikit-learn, TensorFlow/Keras LSTMs), feature preprocessing (scaling), and various ensembling strategies (average, weighted average, confidence-weighted, voting).
- Modular predictor implementations for different ML frameworks.
- Initial framework for dynamic model reloading in the `PredictionService`.

## Technology Stack

*   Python 3.9+ with extensive use of `asyncio` for asynchronous operations
*   PostgreSQL for persistent storage
*   Event-driven architecture with a custom publish-subscribe system
*   Machine Learning Models: XGBoost, Scikit-learn (e.g., RandomForest), TensorFlow/Keras (for LSTMs).
*   Kraken API integration
*   Key Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `talib`, `websockets`, `aiohttp`, `asyncpg`, `PyYAML`, `joblib`.

## Repository Structure

- `src/gal_friday/` - Core trading system components
  - `core/` - Core event system and communication primitives
  - `execution/` - Order execution components
  - `interfaces/` - Abstract base classes for service contracts (e.g., `PredictorInterface`)
  - `predictors/` - Concrete implementations for different ML model types (XGBoost, scikit-learn, LSTM)
  - `model_training/` - Scripts and utilities for training and retraining ML models (conceptual placeholder)
- `config/` - Configuration files and templates
- `docs/` - Comprehensive documentation
  - `Phase 1 - Requirements Analysis & Planning/`
  - `Phase 3 - Design and refining/`
  - `KrakenAPI/` - API documentation for Kraken integration
- `db/schema/` - Database schema definitions
- `tests/` - Test suite for components
- `scripts/` - Utility scripts

## Machine Learning & Prediction

The `PredictionService` is central to the bot's intelligence. It is designed to:
- Consume features from the `FeatureEngine`.
- Load and manage multiple, diverse ML models concurrently as defined in `config.yaml`.
- Support predictors for different frameworks: XGBoost, Scikit-learn, and TensorFlow/Keras for LSTMs (with PyTorch LSTM support planned).
- Handle model-specific feature preprocessing, such as scaling, via scaler objects loaded alongside models.
- For LSTM models requiring sequence inputs, the `PredictionService` buffers features over time to construct the necessary input sequences.
- Offer various ensembling strategies to combine predictions from multiple models, including simple averaging, weighted averaging (by configuration or model confidence), and voting for classification tasks.
- Publish `PredictionEvent`s containing either individual model predictions or ensembled results.
- Includes a basic framework for dynamic model reloading in response to configuration updates.

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

## Development Tools and Configuration

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Static type checking
- **flake8**: Style guide enforcement
- **pytest**: Testing framework
- **pre-commit**: Git hooks to enforce quality

Configuration for these tools is maintained primarily in **pyproject.toml** with additional configuration in **.flake8** for flake8-specific settings. These standardized configurations help ensure consistent code quality across local development and CI/CD pipelines.

For contributor guidelines and detailed information on setting up the development environment, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Specify your license information here]

## Acknowledgments

- Kraken for providing the API infrastructure
- Open source community for the various libraries and tools used in this project

# TESTING
