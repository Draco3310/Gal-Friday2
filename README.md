# Gal-Friday

## Overview
Project Gal-Friday is an automated cryptocurrency trading bot designed to execute high-frequency scalping and day trading strategies on the Kraken exchange. It leverages AI/ML predictive models (XGBoost, RandomForest, LSTM) to trade specific cryptocurrency pairs, XRP/USD and DOGE/USD, with the primary goal of generating a consistent revenue stream. The system aims to provide a reliable, data-driven approach to trading, operating 24/7 within defined risk parameters.

## Features
- Real-time market data ingestion (L2 order book, OHLCV) via WebSocket.
- Feature engineering, including calculation of technical indicators and order book features.
- Predictive modeling using Machine Learning (XGBoost, RandomForest, LSTM) for price movement forecasts.
- Strategy arbitration and signal generation logic based on model predictions and configurable rules.
- Comprehensive risk management, including pre-trade checks, position sizing, and portfolio-level drawdown limits.
- Automated order execution via Kraken REST and WebSocket APIs (Limit/Market orders).
- Real-time portfolio tracking (balance, positions, equity, P&L).
- Extensive logging and auditing of all system activities, trades, and errors.
- System monitoring with alerting and automated HALT mechanisms for safety.
- Backtesting engine for strategy validation with realistic simulation of trading conditions.
- Paper trading capability using Kraken Sandbox or internal simulation.

## Technologies Used
- **Programming Language:** Python 3.9+
- **Exchange Interface:** Kraken APIs (REST and WebSocket)
- **Machine Learning Models:** XGBoost, RandomForest, LSTM
- **Core Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow or PyTorch (for LSTMs), CCXT (potentially for API interaction), websockets (for WebSocket communication)
- **Databases:** PostgreSQL (for relational data like trades, logs), InfluxDB (for time-series data like market data, predictions)
- **Operating System:** Linux (target deployment environment)

## Getting Started

### Prerequisites
- Python 3.9+
- pip (Python package installer)
- Access to a PostgreSQL instance (v13+ recommended)
- Access to an InfluxDB instance (v2.x+ recommended)
- Kraken API Key and Secret (ensure these are for the correct environment: Sandbox for paper trading, Live for actual trading)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/gal-friday.git # Or use the appropriate URL for this project
    cd gal-friday
    ```

2.  **Create and activate a virtual environment (recommended):**
    -   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
The main configuration for Gal-Friday is managed through a YAML file, typically located at `config/config.yaml`.

1.  **Create your configuration file:**
    Copy the example configuration `config/config.example.yaml` to `config/config.yaml`.
    ```bash
    cp config/config.example.yaml config/config.yaml
    ```

2.  **Edit `config/config.yaml` to set up your environment:**
    Key configurations include:
    -   **Database Connections:** Details for connecting to your PostgreSQL and InfluxDB instances (host, port, user, password, database name).
    -   **Kraken API Credentials:** Your API key and secret.
        *   **Security Note:** It is strongly recommended to manage API keys securely, for example, using environment variables or a dedicated secrets management tool, rather than hardcoding them directly into the configuration file. The `config.example.yaml` might show paths or expect environment variable names. Refer to `ConfigManager` and documentation for the exact secure methods supported.
    -   **Operating Mode:** Set to `live`, `paper`, or `backtest`.
    -   **Model Paths & Parameters:** Paths to your pre-trained machine learning models and any associated parameters.
    -   **Trading Pairs:** Specification of the cryptocurrency pairs to trade (e.g., `XRP/USD`, `DOGE/USD`).
    -   **Risk Management:** Parameters like maximum drawdown, risk per trade, position sizing strategy, etc.
    -   **Logging:** Configuration for log levels, file paths, and database logging.

    Refer to the comments within `config/config.example.yaml` for detailed explanations of each parameter.

### Running the Bot
Once configured, you can run the Gal-Friday bot using the following command from the root directory of the project:

```bash
python -m gal_friday.main --config /path/to/your/config.yaml
```
Replace `/path/to/your/config.yaml` with the actual path to your configuration file (e.g., `config/config.yaml`).

To see available command-line options, including overriding the log level:
```bash
python -m gal_friday.main --help
```

The bot will initialize its components, connect to the exchange and databases, and start its trading operations based on your configuration. Monitor the logs for detailed information about its activity.

## Project Structure
The project is organized into the following main directories:

-   `gal_friday/`: Contains the core application code for the trading bot, including modules for data ingestion, feature engineering, prediction, execution, portfolio management, and risk management.
-   `config/`: Holds configuration files. `config.example.yaml` provides a template, and `config.yaml` (or a user-specified file) contains the active configuration for the bot.
-   `tests/`: Includes unit tests and integration tests to ensure the correctness and reliability of the application components.
-   `docs/`: Contains all project-related documentation, such as requirements specifications (SRS), design documents, research notes, and API documentation.
-   `scripts/`: Provides utility and automation scripts for various tasks like deployment, initial model training, data processing, etc.
-   `db/`: Contains database schema definitions and migration scripts (e.g., for PostgreSQL).
-   `models/`: (Assumed location, to be confirmed by actual usage) This directory is intended to store pre-trained machine learning models used by the prediction service.

## Contributing
Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on how to contribute to the project, including coding standards, testing procedures, and the pull request process.

## License
This project is licensed under the terms of the `LICENSE` file. Please review the license for information on permissions and limitations.
