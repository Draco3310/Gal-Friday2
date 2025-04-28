# Project Gal-Friday

Automated cryptocurrency trading bot (High-Frequency Scalping/Day Trading) using AI/ML prediction models.

## Goal

Develop, deploy, and maintain a sophisticated, automated cryptocurrency trading bot targeting XRP/USD and DOGE/USD on Kraken, aiming for significant income generation ($75k/year target on $100k capital) within defined risk parameters.

## Current Status

Phase 3: Implementation (MVP Focus)

## Technology Stack (MVP)

*   Python 3.9+ (`asyncio`)
*   PostgreSQL
*   InfluxDB
*   XGBoost
*   Kraken API (via `ccxt`)
*   Key Libraries: `pandas`, `numpy`, `scikit-learn`, `websockets`, `aiohttp`, `asyncpg`, `influxdb-client`, `PyYAML`, `APScheduler`, `pypubsub`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Gal-Friday
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment:** Set up PostgreSQL and InfluxDB instances. Copy configuration templates from `config/` (to be created) and populate with your credentials/settings.

## Next Steps

*   Task 3.2: Setup Cloud Infrastructure (MVP) - Provision VM, install DBs.
*   Implement core modules (DataIngestor, FeatureEngine, etc.). 