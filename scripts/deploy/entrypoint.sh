#!/bin/bash
set -e

echo "Starting Gal-Friday Trading System..."

# Function to check if a required environment variable is set
check_env_var() {
    if [ -z "${!1}" ]; then
        echo "ERROR: Required environment variable $1 is not set"
        exit 1
    fi
}

# Check required environment variables
echo "Checking environment variables..."
check_env_var "KRAKEN_API_KEY"
check_env_var "KRAKEN_API_SECRET"
check_env_var "INFLUXDB_TOKEN"

# Optional: Wait for dependencies
if [ -n "$WAIT_FOR_POSTGRES" ]; then
    echo "Waiting for PostgreSQL..."
    while ! curl -s "$POSTGRES_URL" > /dev/null 2>&1; do
        sleep 1
    done
    echo "PostgreSQL is ready"
fi

if [ -n "$WAIT_FOR_REDIS" ]; then
    echo "Waiting for Redis..."
    while ! curl -s "$REDIS_URL" > /dev/null 2>&1; do
        sleep 1
    done
    echo "Redis is ready"
fi

# Run database migrations if needed
if [ -n "$RUN_MIGRATIONS" ]; then
    echo "Running database migrations..."
    python -m gal_friday.db.migrate
fi

# Validate configuration
echo "Validating configuration..."
python -m gal_friday.config_manager --validate

# Check credentials
echo "Validating credentials..."
python -m gal_friday.utils.secrets_manager --validate

# Start health check server in background
echo "Starting health check server..."
python -m gal_friday.health_server &
HEALTH_PID=$!

# Trap to ensure cleanup on exit
trap "kill $HEALTH_PID 2>/dev/null || true" EXIT

# Start the main application
echo "Starting main application..."
exec "$@" 