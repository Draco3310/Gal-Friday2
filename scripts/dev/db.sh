#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source .env

case "${1:-}" in
    migrate)
        echo "Running database migrations..."
        alembic upgrade head
        ;;
    rollback)
        echo "Rolling back one migration..."
        alembic downgrade -1
        ;;
    reset)
        echo "Resetting database..."
        alembic downgrade base
        alembic upgrade head
        ;;
    shell)
        echo "Opening database shell..."
        case "${DATABASE_URL}" in
            postgresql://*)
                psql "${DATABASE_URL}"
                ;;
            sqlite://*)
                sqlite3 "${DATABASE_URL#sqlite:///}"
                ;;
            *)
                echo "Unknown database type"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Usage: $0 {migrate|rollback|reset|shell}"
        exit 1
        ;;
esac
