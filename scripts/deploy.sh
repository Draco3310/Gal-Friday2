#!/bin/bash
# Deployment script for Gal-Friday2
# This script is intended to be run by the CI/CD pipeline to deploy the application to a server
# Usage: ./deploy.sh <environment>

set -e  # Exit on any error

ENVIRONMENT=${1:-development}
echo "Deploying to $ENVIRONMENT environment"

# Pull the latest Docker image
docker pull ghcr.io/$GITHUB_REPOSITORY:latest

# Stop and remove existing containers
docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml down || true

# Start the new containers
docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml up -d

# Run database migrations if needed
# docker-compose exec app python -m alembic upgrade head

echo "Deployment to $ENVIRONMENT completed successfully"

# Run health check
echo "Running health check..."
sleep 10  # Give the application time to start
curl -f http://localhost:8000/health || echo "Health check failed, but continuing..."

echo "Deployment process completed"
