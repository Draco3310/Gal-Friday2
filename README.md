# Gal-Friday2

Enterprise-grade automated cryptocurrency trading system.

## Quick Start

```bash
# Setup development environment
./setup.sh --dev

# Activate environment
source .venv/bin/activate

# Run the application
python -m gal_friday.main --config config/config.yaml
```

## Documentation

- [Architecture Overview](docs/architecture/README.md)
- [API Documentation](docs/api/README.md)
- [Deployment Guide](docs/deployment/README.md)
- [Development Guide](docs/guides/development.md)

## Features

- High-frequency cryptocurrency trading
- ML-based price prediction (XGBoost, LSTM, Random Forest)
- Real-time market data processing
- Risk management and portfolio optimization
- WebSocket and REST API integration with Kraken
- Comprehensive monitoring and alerting
- Paper trading and backtesting modes

## Requirements

- Python 3.11+
- PostgreSQL 15+ (for production)
- InfluxDB 2.7+ (for time series data)
- Redis 7+ (for caching)
- 8GB+ RAM recommended
- GPU optional (for ML acceleration)

## License

Proprietary - All rights reserved
